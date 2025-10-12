from odf.opendocument import OpenDocumentText
from odf.style import Style, TextProperties, ParagraphProperties
from odf.text import H, P, Span

# Create the ODT document
textdoc = OpenDocumentText()

# ========== Define Styles ==========
# Title Style
title_style = Style(name="TitleStyle", family="paragraph")
title_style.addElement(TextProperties(attributes={'fontsize': "20pt", 'fontweight': "bold", 'color': "#1E3A8A"}))
textdoc.styles.addElement(title_style)

# Subtitle Style
subtitle_style = Style(name="SubtitleStyle", family="paragraph")
subtitle_style.addElement(TextProperties(attributes={'fontsize': "14pt", 'fontweight': "bold", 'color': "#2563EB"}))
textdoc.styles.addElement(subtitle_style)

# Normal Text Style
body_style = Style(name="BodyStyle", family="paragraph")
body_style.addElement(TextProperties(attributes={'fontsize': "11pt"}))
body_style.addElement(ParagraphProperties(attributes={'marginbottom': "0.2cm"}))
textdoc.styles.addElement(body_style)

# Header Style
header_style = Style(name="HeaderStyle", family="paragraph")
header_style.addElement(TextProperties(attributes={'fontsize': "13pt", 'fontweight': "bold", 'color': "#0F172A"}))
textdoc.styles.addElement(header_style)

# ========== Title ==========
textdoc.text.addElement(P(stylename=title_style, text="PROJECT: KALPA 2035"))
textdoc.text.addElement(P(stylename=subtitle_style, text="A Kalpa in Motion. From Code to Cosmos."))
textdoc.text.addElement(P(stylename=body_style, text=""))

# ========== Introduction ==========
intro = """Kalpa 2035 is the masterplan of RKLS Universe – a vision to expand technology, purpose, and impact through nine evolving companies founded by Madhu Krishna. From the roots of struggle to the heights of global influence, this is a mission shaped in silence, driven by fire, and built for generations."""
textdoc.text.addElement(P(stylename=body_style, text=intro))

# ========== Companies ==========
companies = {
    "SREEIS RKLS LLP": "Flagship for TradeSphere Global – simplifying UK/EU tariffs and global trade automation.",
    "KrisLynx LLP": "AI/IT company behind SelfMate and FearLink, igniting tomorrow’s tech solutions.",
    "RKLS Groups Pvt. Ltd.": "Parent umbrella company driving governance, control, and expansion.",
    "Yours Home Trust": "Heart-driven non-profit focused on upliftment, education, and shelter.",
    "Laxmi Textiles": "Empowering rural India through traditional textile manufacturing.",
    "KRK Fashions": "A global fashion identity blending heritage and innovation.",
    "RKLS InfraLynx": "Redefining infrastructure, smart logistics, and mobility across India.",
    "RKLS EduDominion": "Modern schooling and educational systems rooted in holistic learning.",
    "FearLink": "World’s first fear-detection AI wearable system for personal safety and wellness."
}

textdoc.text.addElement(P(stylename=body_style, text=""))

for company, desc in companies.items():
    textdoc.text.addElement(P(stylename=header_style, text=company))
    textdoc.text.addElement(P(stylename=body_style, text=desc))
    textdoc.text.addElement(P(stylename=body_style, text=""))

# ========== Vision 2025–2030–2035 ==========
vision = """RKLS Universe shall rise across three waves:
- By 2025: Establish solid product foundations (TradeSphere, FearLink, EduDominion).
- By 2030: Expand across India and begin global traction (Middle East, UK, EU).
- By 2035: Cement RKLS as a universal symbol of intelligent systems, safety, and solutions.

A Kalpa is not just a timeline, it’s a transformation.
This is not a startup story – it’s a legend in progress."""
textdoc.text.addElement(P(stylename=header_style, text="Timeline: The 3-Phase Vision"))
textdoc.text.addElement(P(stylename=body_style, text=vision))

# ========== Save ==========
textdoc.save("kalpa_masterplan.odt")
print("✅ kalpa_masterplan.odt created successfully.")
