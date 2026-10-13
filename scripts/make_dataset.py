"""
Builds the fabricated demo dataset for "Kill the Quote Spreadsheet".

Buyer: Konkanveda Foods Pvt Ltd (fictional FMCG), Pune plant.
Category: corrugated packaging, 30 line items.
Five vendors reply in five different, messy shapes.

Outputs (under data/):
  rfx_sample.json            the RFx the co-pilot would produce (canonical line ids L01..L30)
  history_last_year.csv      last year's awarded prices (used for "same as last year")
  vendor_responses/          the five raw vendor replies
  attachments/               ISO certificates
  ground_truth.json          what a careful human would extract (used ONLY by the eval tab)
"""
import json, math, random, csv, os, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VR = DATA / "vendor_responses"
AT = DATA / "attachments"
for d in (DATA, VR, AT):
    d.mkdir(parents=True, exist_ok=True)

random.seed(42)
USD_INR = 88.5  # rate the dataset was priced with; the app shows its own editable assumption

# ---------------------------------------------------------------- line items
# (id, name, kind, L, W, H, ply, print_colours, annual_qty)
ITEMS = [
    ("L01", "Shipper - Glucose biscuit 12 x 800g", "RSC", 400, 300, 250, 5, 2, 240000),
    ("L02", "Shipper - Cream biscuit family pack x 10", "RSC", 450, 320, 280, 5, 2, 180000),
    ("L03", "Shipper - Namkeen 200g x 24", "RSC", 380, 260, 220, 3, 1, 300000),
    ("L04", "Shipper - Namkeen 400g x 12", "RSC", 400, 280, 240, 3, 1, 220000),
    ("L05", "Shipper - Edible oil 1L bottle x 12", "RSC", 330, 250, 290, 5, 2, 160000),
    ("L06", "Shipper - Edible oil 5L jar x 4", "RSC", 400, 300, 330, 7, 2, 60000),
    ("L07", "Shipper - Edible oil 15L tin x 1", "RSC", 260, 260, 370, 7, 1, 40000),
    ("L08", "Shipper - Spice 100g x 48", "RSC", 360, 240, 200, 3, 1, 150000),
    ("L09", "Shipper - Spice 500g x 20", "RSC", 380, 280, 230, 5, 1, 90000),
    ("L10", "Shipper - Tea 250g x 40", "RSC", 420, 300, 260, 5, 2, 110000),
    ("L11", "Shipper - Tea 1kg x 12", "RSC", 400, 300, 280, 5, 2, 70000),
    ("L12", "Shipper - Instant noodles 70g x 96", "RSC", 480, 340, 300, 3, 2, 260000),
    ("L13", "Shipper - Ketchup 1kg bottle x 12", "RSC", 330, 250, 260, 5, 2, 120000),
    ("L14", "Shipper - Jam 500g jar x 24", "RSC", 340, 230, 170, 5, 1, 80000),
    ("L15", "E-com mailer - Small", "RSC", 250, 200, 100, 3, 0, 200000),
    ("L16", "E-com mailer - Medium", "RSC", 350, 250, 150, 3, 0, 150000),
    ("L17", "E-com mailer - Large", "RSC", 450, 350, 250, 5, 0, 80000),
    ("L18", "Export shipper - Biscuit (4-colour)", "RSC", 420, 310, 260, 7, 4, 30000),
    ("L19", "Shipper - Atta 5kg x 4", "RSC", 420, 300, 300, 5, 1, 100000),
    ("L20", "Shipper - Atta 10kg x 2", "RSC", 450, 320, 320, 7, 1, 45000),
    ("L21", "Retail-ready box - Chocolate (4-colour)", "RSC", 300, 200, 150, 3, 4, 180000),
    ("L22", "Festive gift hamper box (4-colour)", "RSC", 400, 300, 120, 5, 4, 25000),
    ("L23", "Partition - 12 cell, fits L05 oil shipper", "PARTITION", 330, 250, 270, 3, 0, 160000),
    ("L24", "Partition - 4 cell, fits L06 5L jar shipper", "PARTITION", 400, 300, 310, 3, 0, 60000),
    ("L25", "Partition - 12 cell, fits L13 ketchup shipper", "PARTITION", 330, 250, 240, 3, 0, 120000),
    ("L26", "Layer pad 1200 x 1000 (pallet)", "PAD", 1200, 1000, 0, 3, 0, 90000),
    ("L27", "Layer pad 1000 x 800 (pallet)", "PAD", 1000, 800, 0, 3, 0, 60000),
    ("L28", "Die-cut display tray - Biscuit (4-colour)", "TRAY", 380, 280, 80, 5, 4, 50000),
    ("L29", "Shelf-ready PDQ tray - Namkeen", "TRAY", 400, 300, 100, 3, 2, 70000),
    ("L30", "Wrap-around sleeve - Jam 12 jar pack", "SLEEVE", 340, 230, 120, 3, 2, 90000),
]

FLUTE = {3: "B", 5: "BC", 7: "BCB"}
BF = {3: 18, 5: 20, 7: 22}
LINER = {3: 150, 5: 160, 7: 180}
FLUTE_GSM = 120
TAKEUP = 1.4


def total_gsm(ply):
    n_liners = (ply + 1) // 2
    n_flutes = ply // 2
    return n_liners * LINER[ply] + n_flutes * FLUTE_GSM * TAKEUP


def area_m2(kind, L, W, H):
    if kind == "RSC":
        return (2 * L + 2 * W + 35) * (W + H) / 1e6
    if kind == "PARTITION":
        cells = {330: (3, 4), 400: (1, 1)}.get(L, (3, 4))
        long_p, short_p = cells
        return (long_p * L + short_p * W) * H / 1e6
    if kind == "PAD":
        return L * W / 1e6
    if kind == "TRAY":
        return (L + 2 * H + 30) * (W + 2 * H + 30) / 1e6
    if kind == "SLEEVE":
        return (2 * L + 2 * W + 30) * H / 1e6
    raise ValueError(kind)


BASE_RATE = {3: 44.0, 5: 46.0, 7: 48.0}          # INR/kg converted board
PRINT_ADD = {0: 0.0, 1: 0.8, 2: 1.5, 4: 3.6}      # INR per piece, scaled by size

lines = []
for (lid, name, kind, L, W, H, ply, pc, qty) in ITEMS:
    a = area_m2(kind, L, W, H)
    wt = a * total_gsm(ply) / 1000.0
    size_factor = max(0.6, min(1.6, a / 0.7))
    base = wt * BASE_RATE[ply] + PRINT_ADD[pc] * size_factor
    size = f"{L} x {W} x {H} mm" if H else f"{L} x {W} mm"
    spec = f"{ply}-ply, {FLUTE[ply]} flute, {BF[ply]} BF, kraft {LINER[ply]} GSM liners"
    printing = "Plain" if pc == 0 else f"{pc}-colour flexo"
    lines.append(dict(
        line_id=lid, item=name, kind=kind, size_mm=size, L=L, W=W, H=H,
        ply=ply, flute=FLUTE[ply], bf=BF[ply], liner_gsm=LINER[ply],
        printing=printing, annual_qty=qty, uom="pcs",
        est_weight_kg=round(wt, 3), _base=base, spec=spec,
    ))

QUESTIONNAIRE = [
    {"q_id": "Q1", "question": "Do you hold a valid ISO 9001:2015 certificate? Attach a copy.", "gate": True,
     "pass_rule": "Yes, with a certificate that is valid on the bid date"},
    {"q_id": "Q2", "question": "Do you have an in-house lab for BCT, ECT and burst strength testing?", "gate": True,
     "pass_rule": "Yes (in-house, not outsourced)"},
    {"q_id": "Q3", "question": "Monthly converting capacity (MT/month)?", "gate": False, "pass_rule": ""},
    {"q_id": "Q4", "question": "Standard lead time from PO to dispatch (days)?", "gate": True,
     "pass_rule": "15 days or less"},
    {"q_id": "Q5", "question": "Do you use food-grade / low-migration inks for printed boxes?", "gate": False, "pass_rule": ""},
    {"q_id": "Q6", "question": "Recycled fibre content (%) and FSC certification status?", "gate": False, "pass_rule": ""},
    {"q_id": "Q7", "question": "Payment terms offered?", "gate": False, "pass_rule": ""},
    {"q_id": "Q8", "question": "Price validity period?", "gate": False, "pass_rule": ""},
]

TERMS = {
    "delivery": "DAP Konkanveda Foods, Plot 14, MIDC Ranjangaon, Pune 412220",
    "price_basis": "INR per piece, exclusive of GST, inclusive of freight to plant",
    "contract_period": "12 months, call-off against monthly schedule",
    "payment_terms": "60 days from receipt of goods",
    "quote_due": "within 9 days of RFx issue",
}

rfx = {
    "rfx_id": "RFQ-KVF-PKG-2026-031",
    "title": "Annual rate contract - corrugated packaging, Pune plant",
    "buyer": {"company": "Konkanveda Foods Pvt Ltd (fictional)", "name": "Priya Kulkarni", "role": "Category Buyer - Packaging"},
    "category": "Corrugated packaging",
    "currency": "INR",
    "line_items": [{k: v for k, v in l.items() if not k.startswith("_") and k not in ("L", "W", "H")} for l in lines],
    "questionnaire": QUESTIONNAIRE,
    "terms": TERMS,
    "vendors": [
        {"vendor_id": "V1", "name": "Meridian Fold Asia Pte Ltd (India ops, Chennai)", "email": "quotes@meridianfold.example"},
        {"vendor_id": "V2", "name": "Bhimthadi Corrugation Works", "email": "sales@bhimthadi.example"},
        {"vendor_id": "V3", "name": "Ambad Kraftline LLP", "email": "info@ambadkraftline.example"},
        {"vendor_id": "V4", "name": "Krishnai Pack Industries", "email": "krishnaipack@example.com"},
        {"vendor_id": "V5", "name": "Chakan Boxcraft Solutions", "email": "rahul@chakanboxcraft.example"},
    ],
}
(DATA / "rfx_sample.json").write_text(json.dumps(rfx, indent=2))

# ---------------------------------------------------------------- last year's prices
# Chakan Boxcraft (V5) was incumbent on most lines last year; Bhimthadi on a few.
history = []
for l in lines:
    lid = l["line_id"]
    supplier = "Bhimthadi Corrugation Works" if lid in ("L03", "L04", "L08", "L12", "L15", "L16") else "Chakan Boxcraft Solutions"
    last = round(l["_base"] * 0.96 * random.uniform(0.98, 1.03), 2)
    history.append(dict(line_id=lid, item=l["item"], size_mm=l["size_mm"], ply=l["ply"], printing=l["printing"],
                        annual_qty_ly=int(l["annual_qty"] * random.uniform(0.85, 1.0) // 1000 * 1000),
                        awarded_supplier=supplier, unit_price_inr=last, uom="per piece"))
with open(DATA / "history_last_year.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    w.writeheader(); w.writerows(history)
HIST = {h["line_id"]: h for h in history}

# ---------------------------------------------------------------- vendor prices (truth, INR/pc, ex-GST)
def vend_price(base, mult):
    return base * mult * random.uniform(0.95, 1.05)

truth = {"usd_inr_used_to_price": USD_INR, "vendors": {}}
LBY = {l["line_id"]: l for l in lines}

# V1 Meridian: USD per 1000 pcs, own codes, reordered; L09 offered as 3-ply alternative.
v1 = {}
for l in lines:
    p = vend_price(l["_base"], 1.06)
    if l["line_id"] == "L09":
        p = p * 0.78
    v1[l["line_id"]] = p
# V2 Bhimthadi: INR/pc; 2.5% discount on PO value > 75 lakh (footnote); freight included.
v2 = {l["line_id"]: vend_price(l["_base"], 1.00) for l in lines}
# V3 Ambad Kraftline: per box of 100 for RSC/partitions/trays/sleeve, per piece for pads; skips L06, L07, L28.
V3_SKIP = {"L06", "L07", "L28"}
v3 = {l["line_id"]: vend_price(l["_base"], 0.965) for l in lines if l["line_id"] not in V3_SKIP}
# V4 Krishnai: INR/pc on a photographed printed quote; L14 is smudged.
v4 = {l["line_id"]: vend_price(l["_base"], 0.99) for l in lines}
# V5 Global: 42/kg 5-ply, 38/kg 3-ply (RSC boxes), rest same as last year, freight extra.

for vid, prices in (("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4)):
    for k in prices:
        prices[k] = round(prices[k], 2)

# Krishnai prices printed with 2 decimals; round to 0.5 to look like a hand-set rate card
for k in v4:
    v4[k] = round(v4[k] * 2) / 2

# ---------------------------------------------------------------- V1 Meridian - Excel, ignores template
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Commercial Offer"
navy = PatternFill("solid", fgColor="1F3A5F")
pale = PatternFill("solid", fgColor="EAF1F8")
thin = Side(style="thin", color="B7C4D3")
ws["A1"] = "MERIDIAN FOLD ASIA PTE LTD"; ws["A1"].font = Font(size=16, bold=True, color="1F3A5F")
ws["A2"] = "India Operations - Sriperumbudur, Chennai | GSTIN 33AAFCO1234Q1Z9"
ws["A3"] = "Commercial Offer No. OPA/IN/2026/1187  |  Ref: your enquiry for corrugated packaging (Pune)"
ws["A4"] = "All prices in USD per 1000 pcs, EXW Chennai + freight to Pune included in price. GST extra as applicable."
ws["A4"].font = Font(italic=True, color="7A1F1F")
hdr = ["Sr", "Meridian Code", "Product Description", "Board Construction", "Inside Dim. (mm)", "Print", "MOQ (pcs)",
       "Price USD / 1000 pcs", "Lead Time", "Remarks"]
ws.append([]); ws.append(hdr)
for c in range(1, len(hdr) + 1):
    cell = ws.cell(row=6, column=c); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = navy
    cell.alignment = Alignment(wrap_text=True, vertical="center")
order = sorted(lines, key=lambda l: (l["ply"], l["kind"], -l["annual_qty"]))
code_prefix = {"RSC": "CB", "PARTITION": "PT", "PAD": "LP", "TRAY": "DT", "SLEEVE": "WS"}
desc_style = {
    "RSC": lambda l: f"RSC carton for {l['item'].split(' - ', 1)[-1]}",
    "PARTITION": lambda l: f"Partition set {l['item'].split(' - ', 1)[-1].replace(', fits', ' (for')})" if 'fits' in l['item'] else l['item'],
    "PAD": lambda l: f"Layer / tier pad {l['size_mm'].replace(' mm', '')}",
    "TRAY": lambda l: f"Die-cut tray: {l['item'].split(' - ', 1)[-1]}",
    "SLEEVE": lambda l: f"Wraparound for {l['item'].split(' - ', 1)[-1]}",
}
r = 7
v1_rows = []
for i, l in enumerate(order, 1):
    lid = l["line_id"]
    ply = l["ply"]
    board = f"{ply}-Ply {l['flute']} | {l['bf']} BF | {l['liner_gsm']} GSM"
    remark = ""
    if lid == "L09":
        board = "3-Ply B | 22 BF | 200 GSM (alternate)"
        remark = "Offered 3-ply high-BF alternate in place of 5-ply - equivalent BCT, please validate"
    if lid == "L18":
        remark = "Export: price excludes fumigation"
    usd_per_k = round(v1[lid] * 1000 / USD_INR, 1)
    dims = l["size_mm"].replace(" mm", "").replace(" x ", "x")
    row = [i, f"OPA-{code_prefix[l['kind']]}-{4100 + i}", desc_style[l["kind"]](l), board, dims,
           l["printing"].replace("-colour flexo", "C flexo"), 10000 if l["annual_qty"] > 60000 else 5000,
           usd_per_k, "14 days", remark]
    ws.append(row)
    for c in range(1, len(row) + 1):
        cell = ws.cell(row=r, column=c); cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
        if i % 2 == 0: cell.fill = pale
    ws.cell(row=r, column=8).number_format = '#,##0.0'
    v1_rows.append((lid, usd_per_k, f"Commercial Offer!H{r}", remark))
    r += 1
ws.append([])
ws.append(["", "", "Payment: 45 days from invoice. Validity: 30 days. Prices subject to kraft paper index revision quarterly."])
widths = [5, 14, 46, 30, 16, 10, 11, 14, 10, 52]
for i, wdt in enumerate(widths, 1):
    ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = wdt
ws2 = wb.create_sheet("Compliance")
ws2.append(["Item", "Our response"])
for c in (1, 2): ws2.cell(row=1, column=c).font = Font(bold=True)
for row in [
    ["Quality system", "ISO 9001:2015 certified, certificate attached (valid till Mar 2028)"],
    ["Testing", "Fully equipped in-house QC lab: BCT, ECT, bursting strength, Cobb, moisture"],
    ["Capacity", "1,800 MT per month across 2 plants"],
    ["Lead time", "14 days from PO for repeat items; 21 days for new artwork"],
    ["Inks", "Water-based, food-safe inks as standard"],
    ["Sustainability", "Recycled content 65-70%. FSC CoC certified."],
    ["Payment", "45 days from invoice"],
    ["Validity", "30 days"],
]:
    ws2.append(row)
ws2.column_dimensions["A"].width = 18; ws2.column_dimensions["B"].width = 80
wb.save(VR / "V1_Meridian_Fold_commercial_offer.xlsx")

# ---------------------------------------------------------------- V2 Bhimthadi - PDF on letterhead
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

ss = getSampleStyleSheet()
small = ParagraphStyle("small", parent=ss["Normal"], fontSize=6.5, textColor=colors.grey, leading=8)
norm = ParagraphStyle("n", parent=ss["Normal"], fontSize=8.5, leading=11)


def letterhead(c, doc):
    c.saveState()
    c.setFillColor(colors.HexColor("#8B1E1E")); c.rect(0, A4[1] - 28 * mm, A4[0], 28 * mm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 17)
    c.drawString(15 * mm, A4[1] - 14 * mm, "BHIMTHADI CORRUGATION WORKS")
    c.setFont("Helvetica", 8)
    c.drawString(15 * mm, A4[1] - 20 * mm, "Corrugated Boxes | Partitions | Pads   -   J-112, MIDC Bhosari, Pune 411026")
    c.drawString(15 * mm, A4[1] - 24 * mm, "GSTIN 27AADFD5521K1ZP   |   sales@bhimthadi.example   |   +91 20 2712 0000")
    c.setFillColor(colors.grey); c.setFont("Helvetica", 7)
    c.drawString(15 * mm, 10 * mm, f"Bhimthadi Corrugation Works - Quotation DPI/Q/26-27/442 - page {doc.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(VR / "V2_Bhimthadi_quotation.pdf"), pagesize=A4,
                        topMargin=34 * mm, bottomMargin=18 * mm, leftMargin=14 * mm, rightMargin=14 * mm)
story = [Paragraph("<b>QUOTATION</b> &nbsp; No. DPI/Q/26-27/442 &nbsp;&nbsp; Date: 08-Oct-2026", norm), Spacer(1, 4),
         Paragraph("To: Ms. Priya Kulkarni, Category Buyer - Packaging, Konkanveda Foods Pvt Ltd, Ranjangaon. "
                   "Ref: RFQ-KVF-PKG-2026-031. We thank you for the enquiry and are pleased to quote as below.", norm),
         Spacer(1, 6)]
rows = [["Line", "Description", "Size (mm)", "Board", "Print", "Rate (Rs/pc)*"]]
v2_rows = []
for l in lines:
    rows.append([l["line_id"], Paragraph(l["item"], norm), l["size_mm"].replace(" mm", ""),
                 f"{l['ply']}P {l['flute']} {l['bf']}BF", l["printing"].replace(" flexo", ""), f"{v2[l['line_id']]:.2f}"])
    v2_rows.append((l["line_id"], v2[l["line_id"]]))
t = Table(rows, colWidths=[12 * mm, 70 * mm, 30 * mm, 26 * mm, 22 * mm, 22 * mm], repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B1E1E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
    ("ALIGN", (-1, 1), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F0F0")]),
]))
story += [t, Spacer(1, 8)]
story.append(Paragraph("<b>Terms &amp; Conditions</b>", norm))
story.append(Paragraph("1. Prices are ex-works Bhosari inclusive of delivery to your Ranjangaon plant. 2. GST @ 18% extra. "
                       "3. Payment: 60 days. 4. Delivery: 10 days from PO for repeat items. 5. Validity: 45 days.", norm))
story.append(Spacer(1, 10))
story.append(Paragraph("<b>Response to your questionnaire</b>", norm))
qa = [
    ("Q1", "Yes - ISO 9001:2015, TUV certificate enclosed, valid to 30-Jun-2027."),
    ("Q2", "Yes. In-house lab with BCT tester, ECT and burst testers; test certificate with each lot."),
    ("Q3", "900 MT / month."),
    ("Q4", "10 days for repeat items; 15 days for first-time artwork."),
    ("Q5", "Yes, water-based low-migration inks."),
    ("Q6", "Approx. 80% recycled fibre. FSC certification in process (expected Q1 2027)."),
    ("Q7", "60 days, as requested."),
    ("Q8", "45 days."),
]
for q, a in qa:
    story.append(Paragraph(f"{q}: {a}", norm))
story.append(Spacer(1, 18))
story.append(Paragraph("For Bhimthadi Corrugation Works &nbsp;&nbsp; (Authorised Signatory)", norm))
story.append(Spacer(1, 30))
story.append(Paragraph("* Rates are per piece, ex-GST. A special discount of 2.5% shall be applicable on all line rates "
                       "if the cumulative purchase order value placed on us under this enquiry exceeds Rs. 75,00,000 "
                       "(Rupees Seventy Five Lakh only) during the contract period. Rates linked to kraft paper prices; "
                       "variation beyond +/-5% to be mutually discussed.", small))
doc.build(story, onFirstPage=letterhead, onLaterPages=letterhead)

# ---------------------------------------------------------------- V3 Ambad Kraftline - Word, commercials in paragraphs
from docx import Document
from docx.shared import Pt

d = Document()
st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
d.add_heading("Ambad Kraftline LLP", level=1)
d.add_paragraph("Gat No. 221, Ambad MIDC, Nashik 422010  |  info@ambadkraftline.example")
d.add_paragraph("Date: 09/10/2026")
d.add_paragraph("Kind Attn: Ms Priya Kulkarni, Konkanveda Foods Pvt Ltd")
d.add_paragraph("Sub: Our best offer against your enquiry for corrugated boxes")
d.add_paragraph(
    "Dear Madam, greetings from Ambad Kraftline. We have been in corrugation since 2004 and supply leading FMCG "
    "brands in Maharashtra. Please find our rates below. Kindly note that all our rates are per box, where one box "
    "means a bundle of 100 nos, as is our standard practice for dispatch, unless otherwise mentioned. Rates are "
    "exclusive of GST and inclusive of freight up to your Ranjangaon plant.")

def p100(lid):
    return round(v3[lid] * 100, 0)

blocks = []
# biscuit + namkeen + oil
blocks.append(
    f"For your biscuit range, the glucose biscuit 12 x 800g shipper (400x300x250, 5 ply) will be Rs {p100('L01'):,.0f} per box, "
    f"and the cream biscuit family pack shipper (450x320x280, 5 ply) Rs {p100('L02'):,.0f}. The namkeen 200g x 24 carton in "
    f"3 ply comes to Rs {p100('L03'):,.0f} and the 400g x 12 carton Rs {p100('L04'):,.0f}. For edible oil, the 1 litre x 12 shipper "
    f"is Rs {p100('L05'):,.0f}. Regarding the 5 litre jar shipper and the 15 litre tin shipper in 7 ply, we regret we cannot quote "
    f"at present as our 7 ply line is fully booked for the next two quarters.")
blocks.append(
    f"Spice 100g x 48 shipper: Rs {p100('L08'):,.0f}. Spice 500g x 20 (5 ply): Rs {p100('L09'):,.0f}. Tea 250g x 40: "
    f"Rs {p100('L10'):,.0f}, tea 1kg x 12: Rs {p100('L11'):,.0f}. The noodles 70g x 96 box (3 ply, 2 colour) works out to "
    f"Rs {p100('L12'):,.0f}, ketchup 1kg x 12 to Rs {p100('L13'):,.0f}, and jam 500g x 24 to Rs {p100('L14'):,.0f}.")
blocks.append(
    f"Mailers: small Rs {p100('L15'):,.0f}, medium Rs {p100('L16'):,.0f}, large (5 ply) Rs {p100('L17'):,.0f}. Export biscuit "
    f"shipper in 7 ply four colour, which we can manage on a special run, Rs {p100('L18'):,.0f}. Atta 5kg x 4: Rs {p100('L19'):,.0f}; "
    f"atta 10kg x 2 (7 ply): Rs {p100('L20'):,.0f}. Chocolate retail ready box Rs {p100('L21'):,.0f} and the festive hamper box "
    f"Rs {p100('L22'):,.0f}.")
blocks.append(
    f"Partitions are also quoted per box of 100 sets: 12 cell for oil Rs {p100('L23'):,.0f}, 4 cell for 5L jar Rs {p100('L24'):,.0f}, "
    f"12 cell for ketchup Rs {p100('L25'):,.0f}. Layer pads we quote per piece: 1200 x 1000 at Rs {v3['L26']:.2f} each and "
    f"1000 x 800 at Rs {v3['L27']:.2f} each. The namkeen PDQ tray is Rs {p100('L29'):,.0f} per box and the jam wraparound "
    f"sleeve Rs {p100('L30'):,.0f}. The die-cut biscuit display tray needs a new die; we will revert separately on this.")
for b in blocks:
    d.add_paragraph(b)
d.add_paragraph(
    "On your questions: we are ISO 9001:2015 certified (copy attached). Testing of BCT and burst is done through an "
    "approved third-party lab in Nashik on every new artwork. Our capacity is about 600 MT a month. Lead time is "
    "normally 12 days. We use standard inks; food grade inks can be arranged at extra cost. Recycled content is around "
    "90%, no FSC as of now. Payment 30 days preferred, 45 days negotiable. Rates valid for 30 days.")
d.add_paragraph("Hoping for your valued order. Regards, Sandeep Patil, Partner, Ambad Kraftline LLP, 98220 00000")
d.save(VR / "V3_Ambad_Kraftline_offer.docx")

# ---------------------------------------------------------------- V4 Krishnai - angled photo of a printed rate card
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
f_title = ImageFont.truetype(FONT_DIR + "DejaVuSerif-Bold.ttf", 40)
f_body = ImageFont.truetype(FONT_DIR + "DejaVuSansMono.ttf", 23)
f_bold = ImageFont.truetype(FONT_DIR + "DejaVuSansMono-Bold.ttf", 23)
f_small = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 18)

PW, PH = 1500, 1950
paper = Image.new("RGB", (PW, PH), (246, 243, 232))
dr = ImageDraw.Draw(paper)
dr.text((60, 50), "KRISHNAI PACK INDUSTRIES", font=f_title, fill=(30, 30, 90))
dr.text((60, 102), "Plot B-7, Additional MIDC, Satara 415004   Ph: 02162-240000", font=f_small, fill=(60, 60, 60))
dr.line((60, 135, PW - 60, 135), fill=(30, 30, 90), width=3)
dr.text((60, 150), "QUOTATION - Konkanveda Foods, Ranjangaon       Date: 10/10/26", font=f_body, fill=(20, 20, 20))
y = 205
dr.text((60, y), "No  Item                                 Ply  Rate/pc", font=f_bold, fill=(10, 10, 10))
y += 34
dr.line((60, y, PW - 60, y), fill=(80, 80, 80), width=1)
y += 8
v4_rows = []
for i, l in enumerate(lines, 1):
    name = l["item"].replace("Shipper - ", "").replace(" (4-colour)", " 4C")
    name = (name[:35]).ljust(36)
    rate = f"{v4[l['line_id']]:7.2f}"
    dr.text((60, y), f"{i:>2}  {name} {l['ply']:>2}  {rate}", font=f_body, fill=(15, 15, 15))
    v4_rows.append((l["line_id"], v4[l["line_id"]], y))
    y += 46
dr.line((60, y + 4, PW - 60, y + 4), fill=(80, 80, 80), width=1)
dr.text((60, y + 18), "Rates in Rs per piece. GST extra. Freight included to Ranjangaon.", font=f_small, fill=(40, 40, 40))
dr.text((60, y + 44), "Delivery 15-20 days. Payment 30 days.  -- for Krishnai Pack, (sign)", font=f_small, fill=(40, 40, 40))
# smudge the L14 rate (row 14) so the digit is genuinely ambiguous
l14_y = [r for r in v4_rows if r[0] == "L14"][0][2]
rate_x = 60 + int(f_body.getlength(f"{14:>2}  {'x'*36} {5:>2}  "))
box = (rate_x - 6, l14_y - 4, rate_x + 115, l14_y + 32)
region = paper.crop(box).filter(ImageFilter.GaussianBlur(3.2))
paper.paste(region, box)
smear = Image.new("RGBA", paper.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(smear)
sd.ellipse((rate_x - 20, l14_y - 12, rate_x + 135, l14_y + 40), fill=(120, 100, 70, 60))
paper = Image.alpha_composite(paper.convert("RGBA"), smear).convert("RGB")
# a pen tick mark and a coffee ring for realism
dr = ImageDraw.Draw(paper)
dr.line((1340, 300, 1355, 318, 1385, 270), fill=(20, 40, 160), width=4)
dr.ellipse((1150, 1600, 1420, 1860), outline=(170, 130, 80), width=7)

# place paper on a desk and warp
desk = Image.new("RGB", (2000, 2400), (92, 70, 52))
noise = (np.random.RandomState(1).normal(0, 9, (2400, 2000, 3))).astype(np.int16)
desk = Image.fromarray(np.clip(np.asarray(desk, dtype=np.int16) + noise, 0, 255).astype(np.uint8))


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.array(matrix, dtype=float); B = np.array(pb).reshape(8)
    return np.linalg.solve(A, B).tolist()


dst = [(300, 300), (1700, 355), (1765, 2235), (205, 2185)]  # where paper corners land (tilted, keystoned)
src = [(0, 0), (PW, 0), (PW, PH), (0, PH)]
coeffs = find_coeffs(dst, src)
warped = paper.transform(desk.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
mask = Image.new("L", (PW, PH), 255).transform(desk.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
desk.paste(warped, (0, 0), mask)
# uneven lighting
grad = np.linspace(1.05, 0.72, desk.size[0])[None, :, None] * np.linspace(1.0, 0.85, desk.size[1])[:, None, None]
arr = np.clip(np.asarray(desk, dtype=float) * grad, 0, 255).astype(np.uint8)
photo = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1.1)).resize((1500, 1800), Image.LANCZOS)
photo.save(VR / "V4_Krishnai_Pack_ratecard_photo.jpg", quality=72)

# ---------------------------------------------------------------- V5 Chakan Boxcraft - terse email
email = f"""From: Rahul Mehta <rahul@chakanboxcraft.example>
To: Priya Kulkarni <priya.kulkarni@konkanvedafoods.example>
Date: Fri, 16 Oct 2026 19:42:10 +0530
Subject: Re: RFQ-KVF-PKG-2026-031 corrugated packaging

Hi Priya,

Rs 42/kg for the 5-ply, 38 for the 3-ply, rest same as last year, freight extra.
Questionnaire same as last time. ISO got renewed in Jan, cert attached. Lab and lead time no change (12 days).

Rahul
Chakan Boxcraft Solutions, Chakan
Sent from my iPhone
"""
(VR / "V5_Chakan_Boxcraft_email.txt").write_text(email)

# ---------------------------------------------------------------- ISO certificates
def cert(path, company, number, issued, expiry, body="Bureau of Quality Assurance (fictional)"):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    W_, H_ = A4
    c.setStrokeColor(colors.HexColor("#1F5F3A")); c.setLineWidth(6); c.rect(20 * mm, 20 * mm, W_ - 40 * mm, H_ - 40 * mm)
    c.setFont("Helvetica-Bold", 26); c.drawCentredString(W_ / 2, H_ - 60 * mm, "CERTIFICATE")
    c.setFont("Helvetica", 12); c.drawCentredString(W_ / 2, H_ - 72 * mm, "This is to certify that the Quality Management System of")
    c.setFont("Helvetica-Bold", 18); c.drawCentredString(W_ / 2, H_ - 88 * mm, company)
    c.setFont("Helvetica", 12)
    c.drawCentredString(W_ / 2, H_ - 100 * mm, "has been assessed and found to conform to the requirements of")
    c.setFont("Helvetica-Bold", 16); c.drawCentredString(W_ / 2, H_ - 114 * mm, "ISO 9001:2015")
    c.setFont("Helvetica", 11)
    c.drawCentredString(W_ / 2, H_ - 128 * mm, "Scope: Manufacture and supply of corrugated boxes, partitions and pads")
    c.drawString(40 * mm, 90 * mm, f"Certificate No.: {number}")
    c.drawString(40 * mm, 82 * mm, f"Date of issue: {issued}")
    c.drawString(40 * mm, 74 * mm, f"Valid until: {expiry}")
    c.drawString(40 * mm, 60 * mm, f"Issued by: {body}")
    c.save()

cert(AT / "V1_Meridian_ISO9001.pdf", "Meridian Fold Asia Pte Ltd - India Operations", "QMS-IN-77821", "15-Mar-2025", "14-Mar-2028")
cert(AT / "V2_Bhimthadi_ISO9001.pdf", "Bhimthadi Corrugation Works", "TQ-44/9001/1182", "01-Jul-2024", "30-Jun-2027")
cert(AT / "V3_Ambad_ISO9001.pdf", "Ambad Kraftline LLP", "BQA/9001/20231", "01-Apr-2023", "31-Mar-2026")
cert(AT / "V5_Chakan_ISO9001.pdf", "Chakan Boxcraft Solutions", "BQA/9001/30577", "11-Jan-2026", "10-Jan-2029")

# ---------------------------------------------------------------- ground truth (eval only)
gt = {"note": "Human-verified expected extraction, per piece INR at the dataset USD rate. Used only by the eval tab.",
      "usd_inr": USD_INR, "lines": {}}
for l in lines:
    lid = l["line_id"]
    usdk = [r[1] for r in v1_rows if r[0] == lid][0]
    v3pc = None
    if lid in v3:
        v3pc = v3[lid] if lid in ("L26", "L27") else round(p100(lid) / 100, 2)
    row = {
        "V1": round(usdk * USD_INR / 1000, 2),
        "V2": v2[lid],
        "V3": v3pc,
        "V4": v4[lid] if lid != "L14" else None,
    }
    if l["kind"] == "RSC" and l["ply"] in (3, 5):
        row["V5"] = round(l["est_weight_kg"] * (42 if l["ply"] == 5 else 38), 2)
    elif HIST[lid]["awarded_supplier"] == "Chakan Boxcraft Solutions":
        row["V5"] = HIST[lid]["unit_price_inr"]
    else:
        row["V5"] = None
    gt["lines"][lid] = row
gt["expected_flags"] = {
    "V1": ["currency USD", "unit per 1000 pcs", "L09 spec deviation: 3-ply offered for 5-ply"],
    "V2": ["conditional 2.5% discount above Rs 75 lakh (footnote)"],
    "V3": ["unit: per box = 100 nos", "L26/L27 per piece", "L06, L07, L28 not quoted", "ISO certificate expired 31-Mar-2026", "testing outsourced"],
    "V4": ["photo", "L14 illegible", "questionnaire not answered"],
    "V5": ["per kg pricing - needs box weight", "'rest same as last year' resolved from history", "freight extra",
           "lines where Global was not incumbent are unresolved"],
}
(DATA / "ground_truth.json").write_text(json.dumps(gt, indent=2))

total = sum(l["_base"] * l["annual_qty"] for l in lines)
print(f"Lines: {len(lines)}  approx annual spend: Rs {total/1e7:.2f} crore")
for l in lines[:5]:
    print(l["line_id"], l["item"], l["est_weight_kg"], round(l["_base"], 2))
