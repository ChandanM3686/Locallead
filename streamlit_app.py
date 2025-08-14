import asyncio
import requests
import re
import aiohttp
from urllib.parse import urljoin, urlparse
from flask import Flask
import google.generativeai as genai
import streamlit as st
import json


MAPS_API_KEY = "AIzaSyDrLHe1dTTc78XG7lznF0fWF0o5uKR6HXA"
GENAI_API_KEY = "AIzaSyDyqTVoAQAr3JulygFdYmMoqnQRSgK-8GA"

# --- Initialization and Validation ---
if MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY" or GENAI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.error("API keys are not configured. Please paste your actual keys into the streamlit_app.py file.")
    st.stop()

try:
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash") # Using a powerful model for better suggestions
except Exception as e:
    st.error(f"Failed to configure the Generative AI model. Please check your GENAI_API_KEY. Error: {e}")
    st.stop()

app = Flask(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# --- Helper Functions ---
def ensure_absolute_url(base: str, link: str) -> str:
    """Ensure a URL is absolute, using the base URL if it's relative."""
    if not link:
        return ""
    if base and not urlparse(base).scheme:
        base = "https://" + base
    if urlparse(link).scheme:
        return link
    return urljoin(base.rstrip("/") + "/", link.lstrip("/"))

def domain_is_valid(domain_part: str) -> bool:
    """Check if the domain part of an email address appears valid."""
    if not domain_part:
        return False
    return bool(re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", domain_part))

def run_async(coro):
    """Helper to run async code from the sync Streamlit environment."""
    return asyncio.run(coro)

# -----------------------------
# Asynchronous Scrapers & AI Functions
# -----------------------------
async def get_pages_to_scan(session, url: str) -> list[str]:
    """Analyzes a website to find the most relevant pages for contact info."""
    prioritized_urls, fallback_urls = set(), set()
    try:
        if not urlparse(url).scheme: url = "https://" + url
        async with session.get(url, timeout=10, headers=HEADERS, ssl=False) as response:
            if response.status != 200: return [url]
            html = await response.text()
            all_links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
            for link in all_links:
                if any(kw in link for kw in ["contact", "kontakt"]):
                    prioritized_urls.add(ensure_absolute_url(url, link))
                elif any(kw in link for kw in ["about", "imprint", "legal"]):
                    fallback_urls.add(ensure_absolute_url(url, link))
            if prioritized_urls: return list(prioritized_urls)[:3]
            return list(dict.fromkeys([url] + list(fallback_urls)))[:3]
    except Exception:
        return [url]

async def extract_emails_from_website(session, url):
    if not url: return []
    urls_to_scan = await get_pages_to_scan(session, url)
    all_emails = set()
    for page_url in urls_to_scan:
        try:
            async with session.get(page_url, timeout=10, headers=HEADERS, ssl=False) as resp:
                if resp.status != 200: continue
                html = await resp.text()
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                for e in re.findall(email_pattern, html):
                    if not any(ext in e.lower() for ext in [".png", ".jpg", ".gif", ".webp", ".svg"]):
                        if "@" in e and domain_is_valid(e.split("@")[1]):
                            all_emails.add(e)
        except Exception as e:
            print(f"Could not extract emails from {page_url}: {e}")
    return list(all_emails)


async def extract_social_media(session, url):
    if not url: return {}
    urls_to_scan = await get_pages_to_scan(session, url)
    social_handles = {}
    social_patterns = {
        "Facebook": r'facebook\.com/([\w.\-]+)', "Twitter": r'(?:twitter|x)\.com/([\w_]+)',
        "Instagram": r'instagram\.com/([\w._\-]+)', "LinkedIn": r'linkedin\.com/(?:company|in)/([\w\-]+)',
        "YouTube": r'youtube\.com/(?:user|channel|c)/([\w\-@]+)',
    }
    for page_url in urls_to_scan:
        try:
            async with session.get(page_url, timeout=10, headers=HEADERS, ssl=False) as resp:
                if resp.status != 200: continue
                html = await resp.text()
                for platform, pattern in social_patterns.items():
                    if platform not in social_handles:
                        if match := re.search(pattern, html, re.IGNORECASE):
                            handle = match.group(1)
                            base_url = f"https://www.{platform.lower()}.com/"
                            if platform == "LinkedIn": base_url = "https://www.linkedin.com/company/"
                            social_handles[platform] = ensure_absolute_url(base_url, handle)
        except Exception as e:
            print(f"Could not extract social media from {page_url}: {e}")
    return social_handles

async def analyze_business_nature(company_name, website):
    if not website: return ""
    prompt = f"Describe the business category for '{company_name}' ({website}). Be concise (e.g., 'Italian Restaurant', 'Digital Marketing Agency'). Max 10 words."
    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing business nature: {e}")
        return ""

# --- NEW FEATURE: AI-Powered Target Suggestion ---
async def suggest_targets_with_ai(product_info: str):
    """Uses Gemini to suggest target industries and locations based on a product description."""
    if not product_info:
        return None
    prompt = f"""
    Based on the following product/service description, please suggest potential B2B target markets.
    
    Product/Service Description: "{product_info}"
    
    Please provide your answer in a clean JSON format. The JSON object should have two keys:
    1. "industries": A list of 3-5 specific target industry strings (e.g., "Software Development Companies", "Italian Restaurants", "Dental Clinics").
    2. "locations": A list of 3-5 suitable target locations (e.g., "San Francisco, United States", "London, United Kingdom").
    
    Example Response:
    {{
        "industries": ["Boutique Coffee Shops", "Independent Bookstores", "Artisan Bakeries"],
        "locations": ["Seattle, United States", "Portland, United States", "Melbourne, Australia"]
    }}
    
    Your Response:
    """
    try:
        response = await model.generate_content_async(prompt)
        # Clean up the response to extract only the JSON part
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Failed to get AI suggestions. Error: {e}")
        return None

async def fetch_leads(industry, location, city, search_type, radius):
    search_query = f"{industry} in {city}, {location}" if city and location else f"{industry} in {city or location}"
    
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": search_query, "key": MAPS_API_KEY}

    if search_type == "Radius" and radius:
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geocode_params = {"address": f"{city}, {location}" if city else location, "key": MAPS_API_KEY}
        try:
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            if geocode_data := geocode_response.json().get("results"):
                loc = geocode_data[0]["geometry"]["location"]
                base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                params = {
                    "location": f"{loc['lat']},{loc['lng']}", "radius": int(float(radius) * 1000),
                    "keyword": industry, "key": MAPS_API_KEY
                }
        except Exception as e:
            print(f"Geocoding failed, falling back to text search. Error: {e}")
    
    response = requests.get(base_url, params=params, timeout=20)
    response.raise_for_status()
    places = response.json().get("results", [])

    async with aiohttp.ClientSession() as session:
        tasks = [process_single_place(session, place, industry) for place in places[:15]]
        return [lead for lead in await asyncio.gather(*tasks) if lead]

async def process_single_place(session, place, industry):
    place_id = place.get("place_id")
    if not place_id: return None

    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    details_params = {
        "place_id": place_id, "fields": "name,formatted_address,website,formatted_phone_number",
        "key": MAPS_API_KEY
    }
    try:
        async with session.get(details_url, params=details_params, timeout=10, ssl=False) as resp:
            details = (await resp.json()).get("result", {})
    except Exception: return None

    website = details.get("website")
    company_name = details.get("name", "")
    
    emails, social_handles, business_nature = [], {}, industry
    if website:
        tasks = [
            extract_emails_from_website(session, website),
            extract_social_media(session, website),
            analyze_business_nature(company_name, website)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        emails = results[0] if not isinstance(results[0], Exception) else []
        social_handles = results[1] if not isinstance(results[1], Exception) else {}
        business_nature = results[2] if not isinstance(results[2], Exception) else industry

    return {
        "Company Name": company_name, "Nature of Business": business_nature,
        "Email IDs": emails, "Contact Numbers": details.get("formatted_phone_number", ""),
        "Social Media Handles": social_handles, "Address": details.get("formatted_address", ""),
        "Website": website or "",
    }

# -----------------------------
# Streamlit App UI
# -----------------------------
def main():
    st.set_page_config(layout="wide", page_title="B2B Lead Generator")

    st.title("🚀 B2B Lead Generator")
    st.markdown("Enter your target criteria, or describe your product to get AI-powered suggestions.")

    col1, col2 = st.columns(2)

    with col1:
        product_info = st.text_area("📦 Your Product/Service Info", 
                                    placeholder="Describe what you are selling to get AI-powered target suggestions...",
                                    height=150)
        
        # --- NEW: Suggest Targets Button ---
        if st.button("🤖 Get AI Target Suggestions"):
            if product_info:
                with st.spinner("🧠 Gemini is thinking of the best targets for you..."):
                    suggestions = run_async(suggest_targets_with_ai(product_info))
                    st.session_state['suggestions'] = suggestions
            else:
                st.warning("Please describe your product/service first.")
        
        # --- NEW: Display Suggestions in an Expander (acts like a popup) ---
        if 'suggestions' in st.session_state and st.session_state['suggestions']:
            with st.expander("🎯 AI Target Suggestions", expanded=True):
                suggestions = st.session_state['suggestions']
                st.write("**Suggested Industries:**")
                st.write(", ".join(suggestions.get("industries", ["N/A"])))
                st.write("**Suggested Locations:**")
                st.write(", ".join(suggestions.get("locations", ["N/A"])))
                st.info("You can now enter these suggestions into the fields below or use your own.")


    with col2:
        industry = st.text_input("🎯 Target Industry", placeholder="e.g., Software Development, Restaurants")
        location = st.text_input("🌍 Country", placeholder="e.g., United States")
        city = st.text_input("🏙️ City", placeholder="e.g., San Francisco")
        
    search_type_col, radius_col = st.columns(2)
    with search_type_col:
        search_type = st.selectbox("Search Area Type", ["City", "Radius"], index=0, help="**City**: Searches the general area. **Radius**: Searches within a specific distance from the city center.")
    with radius_col:
        if search_type == "Radius":
            radius = st.slider("Radius (in Kilometers)", 1, 50, 10)
        else:
            radius = None # Explicitly set to None when not used

    if st.button("Generate Leads", type="primary", use_container_width=True):
        if not all([industry, (location or city)]):
            st.warning("Please fill in all required fields: Industry, and at least a Country or City.")
        else:
            with st.spinner("Finding leads... This may take a moment..."):
                try:
                    leads_result = run_async(
                        fetch_leads(industry, location, city, search_type, radius)
                    )
                    st.session_state['leads'] = leads_result # Save results to session state
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    import traceback
                    traceback.print_exc()
    
    # --- Display Results ---
    if 'leads' in st.session_state:
        leads_result = st.session_state['leads']
        if not leads_result:
            st.info("No leads found. Try broadening your search criteria.")
        else:
            st.success(f"🎉 Found {len(leads_result)} potential leads!")
            for i, lead in enumerate(leads_result):
                st.markdown("---")
                st.subheader(f"{i+1}. {lead.get('Company Name', 'Unknown Company')}")
                
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**Business Nature:** {lead.get('Nature of Business', 'N/A')}")
                    st.markdown(f"**Address:** {lead.get('Address', 'N/A')}")
                    if website := lead.get('Website'): st.markdown(f"**Website:** [{website}]({website})")

                with info_col2:
                    st.markdown(f"**Phone:** {lead.get('Contact Numbers', 'N/A')}")
                    emails = lead.get("Email IDs", [])
                    st.markdown(f"**Emails:** {', '.join(emails) if emails else 'Not found'}")
                    if socials := lead.get("Social Media Handles"):
                        social_links = [f"[{p}]({l})" for p, l in socials.items()]
                        st.markdown(f"**Social Media:** {' | '.join(social_links)}")

if __name__ == "__main__":
    main()