import streamlit as st

st.set_page_config(page_title="Model Glossary", page_icon="📖", layout="wide")

st.markdown("""
<style>
.stTabs [aria-selected="true"] {
    color: #8B5E3C !important;
    border-bottom-color: #8B5E3C !important;
}
hr { border-color: #C4A882 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📖 Model Glossary")
st.markdown(
    "A reference guide to every variable the price predictor uses. Features are grouped by how they are created. "
    "For each one you can see where the data comes from, how it is processed, and why it was included in the model."
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Raw & Structured",
    "Engineered Features",
    "Text & NLP Flags",
    "Interaction Features",
])

with tab1:
    st.subheader("Raw & Structured Features")
    st.caption("These come directly from the auction listing with minimal transformation — mostly cleaning and type conversion.")

    features = [
        {
            "Feature": "Make",
            "Type": "Categorical",
            "Encoding": "Target Encoded",
            "Description": "The car manufacturer (e.g. BMW, Porsche, Toyota). Because there are hundreds of makes with highly unequal representation, it is target encoded — each make is replaced with its smoothed average log-price rather than one-hot encoded, which would create hundreds of sparse columns.",
        },
        {
            "Feature": "Model",
            "Type": "Categorical",
            "Encoding": "Target Encoded",
            "Description": "The specific model name within a make (e.g. 3 Series, 911, Land Cruiser). Also target encoded for the same high-cardinality reason as Make.",
        },
        {
            "Feature": "Mileage",
            "Type": "Numeric",
            "Encoding": "Raw (cleaned)",
            "Description": "Odometer reading in miles. The raw string (e.g. '53,700') is cleaned by stripping commas and converting to a float. Missing mileage is imputed with the column median during training.",
        },
        {
            "Feature": "Engine Displacement",
            "Type": "Numeric",
            "Encoding": "Normalized to liters",
            "Description": "Engine size extracted from the Engine field. The raw text can list displacement in liters (3.0L), cubic centimeters (3000cc), or cubic inches (183ci) — all are normalized to liters so the model sees a single consistent numeric scale.",
        },
        {
            "Feature": "Gears",
            "Type": "Numeric",
            "Encoding": "Extracted via regex",
            "Description": "Number of forward gears extracted from the Transmission field using a regex for patterns like '6-speed'. CVTs have no discrete gear count and are assigned a sentinel value of 1.0 so the column stays numeric and CVT remains clearly distinct from multi-speed automatics.",
        },
        {
            "Feature": "Transmission Type",
            "Type": "Binary",
            "Encoding": "One-hot (Automatic flag)",
            "Description": "Whether the car has an automatic or manual transmission. Cleaned to enforce exactly three valid values — Automatic, Manual, or Other — to filter out data entry noise. The model receives a single binary flag for Automatic (1) vs not (0).",
        },
        {
            "Feature": "S&P 500 Close",
            "Type": "Numeric",
            "Encoding": "Raw (live fetch)",
            "Description": "The S&P 500 closing price on the day of the auction, fetched live via the yfinance library. Included as a macro-economic context signal — auction results from 2021 (bull market) look very different from 2023 (higher rates). Falls back to 5,000 if the API is unavailable at inference time.",
        },
        {
            "Feature": "Exterior Color",
            "Type": "Categorical",
            "Encoding": "Label Encoded",
            "Description": "Simplified from hundreds of manufacturer-specific color names (e.g. 'Estoril Blue', 'Midnight Black') down to ~15 standard color groups (Blue, Black, Gray, etc.) using a lookup table with special-case handling for tricky names like 'Titanium' → Gray and 'Midnight' → Black.",
        },
        {
            "Feature": "Interior Color",
            "Type": "Categorical",
            "Encoding": "Label Encoded",
            "Description": "Same color normalization applied to the interior. Encoded separately from exterior color since interior color carries its own independent price signal (e.g. tan or red leather on a sports car vs black cloth).",
        },
        {
            "Feature": "Title Status",
            "Type": "Categorical",
            "Encoding": "One-hot",
            "Description": "The legal title status of the car. Raw values have dozens of rare variants (e.g. 'Rebuilt/Salvage', 'Reconstructed/Totaled') that are bucketed into ~6 groups: Clean, Rebuilt/Salvage, Mileage Issue, Buyback, Alternate Doc, and Other. A clean title is by far the most common.",
        },
        {
            "Feature": "Drivetrain",
            "Type": "Categorical",
            "Encoding": "One-hot",
            "Description": "Front-wheel drive, rear-wheel drive, all-wheel drive, or four-wheel drive. One-hot encoded with RWD as the reference category since it is the most common on this platform.",
        },
        {
            "Feature": "Seller Type",
            "Type": "Categorical",
            "Encoding": "Used in interaction feature",
            "Description": "Private Party or Dealer. Raw values have many dealer fee variants (e.g. 'Dealer ($95 Doc Fee)') that are all collapsed to 'Dealer'. Not used as a standalone feature — instead it is combined with Title Status into an interaction term.",
        },
    ]

    for f in features:
        with st.expander(f"**{f['Feature']}** — {f['Type']} · {f['Encoding']}"):
            st.markdown(f['Description'])

with tab2:
    st.subheader("Engineered Features")
    st.caption("These do not exist in the raw listing — they are derived or computed during preprocessing to give the model richer signals.")

    features2 = [
        {
            "Feature": "Car Age",
            "Type": "Numeric",
            "How": "auction_year − model_year",
            "Description": "How old the car was at the time of the auction. The model year is extracted from the opening sentence of the Highlights field (Cars & Bids listings always start with 'THIS... is a YYYY...') rather than a structured field, because the Year column in the raw data was sometimes unreliable. Clamped to 0 for pre-production or early-release models.",
        },
        {
            "Feature": "Model Year",
            "Type": "Numeric",
            "How": "Regex on Highlights text",
            "Description": "The car's production year, parsed from the listing's opening sentence pattern 'THIS… is a YYYY'. Falls back to auction year when the pattern is not found, making car age 0 rather than missing. Kept as a separate feature from car age because it carries its own signal — a 1990 car and a 2010 car that are both 10 years old at auction behave differently.",
        },
        {
            "Feature": "Mileage Per Year",
            "Type": "Numeric",
            "How": "Mileage ÷ (car_age + 0.5)",
            "Description": "Average annual mileage — a normalization that makes mileage comparable across cars of different ages. A 50,000 mile car that is 5 years old (10k/yr) is very different from a 50,000 mile car that is 20 years old (2.5k/yr). The +0.5 prevents division by zero for brand-new cars.",
        },
        {
            "Feature": "Flaw Count",
            "Type": "Numeric",
            "How": "Count of comma-separated items in Known Flaws",
            "Description": "A simple count of the number of distinct flaws the seller disclosed. More items in the Known Flaws field generally indicates a more problematic car. Used alongside flaw severity score to capture both breadth and severity of disclosed issues.",
        },
        {
            "Feature": "Flaw Severity Score",
            "Type": "Numeric",
            "How": "Weighted regex scoring across Known Flaws and Seller Notes",
            "Description": (
                "An additive penalty score computed from keyword matches across Known Flaws and Seller Notes. "
                "Penalties are tiered by estimated repair cost severity:\n\n"
                "- **Catastrophic (30–50 pts):** salvage/rebuilt title, flood damage, frame damage, true mileage unknown, rust holes, engine knock, head gasket\n"
                "- **Expensive (5–10 pts):** fluid leaks, cracks/tears, dents, paint chips\n\n"
                "Higher scores push the predicted price down. A car with frame damage and a TMU odometer would score 90+ before any other factors."
            ),
        },
        {
            "Feature": "Trim Tier",
            "Type": "Categorical",
            "How": "Token matching on URL slug",
            "Description": (
                "The car's trim level bucketed into one of six price tiers, derived by parsing the auction URL slug "
                "(e.g. '.../2006-porsche-996-911-gt3' → 'gt3' → ultra_premium). Tiers are:\n\n"
                "- **ultra_premium:** GT3, STO, G63, Evo, Spider (~$200K+ avg)\n"
                "- **high_performance:** Turbo, GTS, AMG, Z06, RS, Carrera 4S (~$100–200K)\n"
                "- **sport_premium:** Shelby, GT500, SRT, Hellcat, Plaid, Raptor (~$70–100K)\n"
                "- **base:** Standard production models (catch-all)\n"
                "- **economy:** Club, Standard, RWD, MX (~$40K)\n"
                "- **unknown:** No URL available; target encoder falls back to the mean"
            ),
        },
        {
            "Feature": "Auction Year / Month",
            "Type": "Numeric",
            "How": "Extracted from Auction Date",
            "Description": "The calendar year and month of the auction. Year captures long-run platform growth and market trends (prices rose sharply in 2021–2022). Month captures seasonal patterns — convertibles sell differently in January vs June.",
        },
        {
            "Feature": "State",
            "Type": "Categorical",
            "Encoding": "Target Encoded",
            "Description": "The US state where the car is located, parsed from the seller's listed location (e.g. 'San Diego, CA 92101' → 'CA'). California and other dry-climate states command a premium for rust-free cars. Target encoded due to high cardinality.",
        },
    ]

    for f in features2:
        with st.expander(f"**{f['Feature']}** — {f['Type']} · {f['How']}"):
            st.markdown(f['Description'])

with tab3:
    st.subheader("Text & NLP Flags")
    st.caption(
        "Two categories: binary keyword flags triggered by exact regex matches in the listing text, "
        "and latent text components from a TF-IDF + SVD pipeline applied to the full listing description."
    )

    st.write("#### Keyword Indicator Flags")
    st.markdown("Each flag is a binary 0/1 field. A value of 1 means the relevant keyword pattern was found in the specified listing section.")

    flags = [
        ("2 Keys Included", "Other Items Included in Sale", "Matches '2 keys', '3 keys', 'both original keys', or 'both keys'. Indicates the seller has both original keys, which matters especially for modern cars with expensive key programming."),
        ("Owner's Manual Included", "Other Items Included in Sale", "Matches 'manual' or 'owners'. Indicates the original owner's manual is included — a sign of a well-preserved, document-complete car."),
        ("Dry Climate / Rust Free", "All listing fields", "Matches 'california', 'arizona', 'texas', 'nevada', 'dry state', or 'no rust'. Cars from low-humidity states command a premium because they are significantly less likely to have rust."),
        ("Project Car", "All listing fields", "Matches 'project', 'needs restoration', 'not running', or 'tow away'. Flags cars being sold as incomplete or non-running, which significantly lowers expected price."),
        ("New Tires", "Recent Service History", "Matches 'new tires', 'replaced tires', 'michelin', or 'fresh rubber'. A recent tire replacement is a meaningful service item that signals the seller invested in the car."),
        ("Sport Seats", "All listing fields", "Matches 'recaro', 'sport seat', 'bucket seat', 'vader', or 'wingback'. Sport seats are a notable upgrade that correlates with performance-oriented builds."),
        ("Emissions Mentioned", "Seller Notes", "Matches 'emissions'. Sellers who explicitly mention passing emissions or smog checks tend to be selling newer or more compliant cars."),
        ("Loan / Financing Mentioned", "Seller Notes", "Matches 'loan'. Indicates the seller mentions an existing loan or financing, which can complicate the sale process."),
        ("Single Owner", "All listing fields", "Matches 'one owner'. Note: single-owner cars on this platform skew older and therefore show a lower average price — this is a confounding effect, not evidence that single ownership hurts value."),
        ("Carfax Mentioned", "Known Flaws", "Matches 'carfax'. Sellers who reference a Carfax report in the flaws section are typically disclosing a reported incident."),
        ("Recent Major Service", "Recent Service History", "Matches 'timing belt', 'clutch', 'ims', 'head gasket', 'water pump', 'transmission replaced', or 'engine replaced'. These are high-cost maintenance items — having them recently done adds real value."),
        ("Modification Status", "Modifications field", "Categorical: 'stock' (no mods), 'light_mod' (exhaust, wheels, coilovers, intake), 'heavy_mod' (turbo kit, engine swap, roll cage), or 'unknown_mod'. Heavy modifications can increase or decrease value depending on the buyer pool."),
    ]

    for name, field, desc in flags:
        with st.expander(f"**{name}** — sourced from: *{field}*"):
            st.markdown(desc)

    st.write("#### Listing Description Text")
    st.markdown(
        "All seller-written text fields (Highlights, Equipment, Modifications, Known Flaws, Recent Service History, "
        "Ownership History, Seller Notes, Other Items) are concatenated into a single document per listing. "
        "This is then vectorized using **TF-IDF** (which down-weights words that appear in almost every listing) "
        "and compressed into **20 latent dimensions using SVD** (Singular Value Decomposition). "
        "Each dimension captures a different axis of variation in how sellers describe their cars — for example, "
        "one dimension might separate track-focused listings from comfort-focused ones, while another separates "
        "well-documented cars from sparse listings. These 20 components collectively appear as a single "
        "**'Listing Description Text'** bar in the SHAP chart."
    )

with tab4:
    st.subheader("Interaction Features")
    st.caption("These combine two existing features into one to let the model capture effects that neither variable can express alone.")

    interactions = [
        {
            "Feature": "Make × Model × Year (make_model_year)",
            "Encoding": "Target Encoded",
            "Description": "A string key combining make, model, and model year (e.g. 'Porsche_996 911_1999'). Target encoded to a single number representing the smoothed average log-price for that exact generation. This is the most powerful feature in the model — a 1999 Porsche 911 and a 2005 Porsche 911 are completely different cars in completely different markets, which neither Make, Model, nor Year alone can capture.",
        },
        {
            "Feature": "Make × Model × Mileage Bucket (make_model_mileage_bucket)",
            "Encoding": "Target Encoded",
            "Description": "Combines make, model, and a mileage decile bucket (0–10%, 10–20%, etc.) into a single target-encoded key. Captures the fact that mileage penalizes different models differently — a high-mileage Ferrari is penalized more severely than a high-mileage Honda because Ferrari buyers are more sensitive to odometer readings.",
        },
        {
            "Feature": "Make × Model × Trim (make_model_trim)",
            "Encoding": "Target Encoded",
            "Description": "Combines make, model, and trim tier into a target-encoded key. Allows the model to price a Porsche 911 GT3 (ultra_premium) very differently from a base 911 Carrera, even within the same make and model.",
        },
        {
            "Feature": "Seller Type × Title Status (seller_x_title)",
            "Encoding": "Target Encoded",
            "Description": "Combines seller type (Private Party / Dealer) with title status (Clean, Rebuilt/Salvage, etc.). Captures asymmetries — for example, a dealer selling a rebuilt-title car may price it differently than a private party in the same situation, and buyers may respond differently to each combination.",
        },
        {
            "Feature": "Car Age × Mileage",
            "Encoding": "Raw numeric product",
            "Description": "The product of car age and mileage. Captures a joint deterioration signal that neither variable expresses on its own: a 20-year-old car with 150,000 miles has a compounding wear effect that is worse than the sum of its parts. The model can still use car age and mileage independently — this term gives it an explicit signal for the interaction.",
        },
        {
            "Feature": "S&P 500 × Auction Year",
            "Encoding": "Raw numeric product",
            "Description": "The product of the S&P 500 close price and the auction year. Captures the combined effect of market conditions and time period — not just whether the market is up, but whether it is up *now* vs several years ago when the platform and its buyer pool were very different.",
        },
    ]

    for f in interactions:
        with st.expander(f"**{f['Feature']}** — {f['Encoding']}"):
            st.markdown(f['Description'])
