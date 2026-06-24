"""
Generate SEAL-DSA Phase 2 PowerPoint Presentation
Run: python generate_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ── Colors ──────────────────────────────────────────────────
C_BG       = RGBColor(0x0F, 0x17, 0x2A)   # dark navy
C_BG2      = RGBColor(0x1E, 0x29, 0x3B)   # card bg
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_PURPLE   = RGBColor(0x63, 0x66, 0xF1)   # primary
C_CYAN     = RGBColor(0x06, 0xB6, 0xD4)   # accent
C_GREEN    = RGBColor(0x10, 0xB9, 0x81)   # green
C_AMBER    = RGBColor(0xF5, 0x9E, 0x0B)   # amber
C_RED      = RGBColor(0xEF, 0x44, 0x44)   # red
C_MUTED    = RGBColor(0x94, 0xA3, 0xB8)   # slate
C_LIGHT    = RGBColor(0xE2, 0xE8, 0xF0)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]  # completely blank

def add_slide():
    return prs.slides.add_slide(BLANK)

def bg(slide, color=C_BG):
    """Fill slide background."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def box(slide, x, y, w, h, bg_color=None, border_color=None, border_width=Pt(0), radius=False):
    """Add a rectangle shape."""
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.line.width = border_width
    if border_color:
        shape.line.color.rgb = border_color
    else:
        shape.line.fill.background()
    if bg_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
    else:
        shape.fill.background()
    return shape

def txt(slide, text, x, y, w, h, size=18, bold=False, color=C_WHITE,
        align=PP_ALIGN.LEFT, wrap=True, italic=False):
    """Add a text box."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return tb

def multiline_txt(slide, lines, x, y, w, h, default_size=16,
                  default_color=C_MUTED, align=PP_ALIGN.LEFT):
    """
    lines = list of dicts: {text, size, bold, color, align}
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = line.get('align', align)
        p.space_before = Pt(line.get('space_before', 0))
        run = p.add_run()
        run.text = line.get('text', '')
        run.font.size = Pt(line.get('size', default_size))
        run.font.bold = line.get('bold', False)
        run.font.italic = line.get('italic', False)
        run.font.color.rgb = line.get('color', default_color)
    return tb

def accent_bar(slide, color=C_PURPLE):
    """Top accent gradient bar."""
    s1 = box(slide, 0, 0, 4.44, 0.055, bg_color=C_PURPLE)
    s2 = box(slide, 4.44, 0, 4.44, 0.055, bg_color=C_CYAN)
    s3 = box(slide, 8.88, 0, 4.45, 0.055, bg_color=C_GREEN)

def footer(slide, num, total=14):
    txt(slide, "SEAL-DSA — Phase 2 Presentation", 0.3, 7.2, 8, 0.3,
        size=9, color=C_MUTED, align=PP_ALIGN.LEFT)
    txt(slide, f"{num} / {total}", 12.0, 7.2, 1.0, 0.3,
        size=9, color=C_MUTED, align=PP_ALIGN.RIGHT)

def card(slide, x, y, w, h, accent_color=C_PURPLE, title="", body_lines=None):
    box(slide, x, y, w, h, bg_color=C_BG2)
    box(slide, x, y, 0.05, h, bg_color=accent_color)
    if title:
        txt(slide, title, x+0.15, y+0.1, w-0.2, 0.35, size=14, bold=True, color=C_LIGHT)
    if body_lines:
        multiline_txt(slide, body_lines, x+0.15, y+0.5, w-0.25, h-0.6, default_size=12)

def divider(slide, y, color=C_BG2):
    box(slide, 0.3, y, 12.73, 0.02, bg_color=color)


# ════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl)
accent_bar(sl)

# Gradient overlay
box(sl, 0, 0, 13.33, 7.5, bg_color=RGBColor(0x0A, 0x0E, 0x1F))

# Glowing circle decoration
box(sl, 7.5, 0.5, 5, 5, bg_color=RGBColor(0x1A, 0x1A, 0x3E))

# Tag
b = box(sl, 3.5, 1.0, 6.3, 0.45, bg_color=RGBColor(0x1E, 0x29, 0x3B))
b.line.color.rgb = C_PURPLE
b.line.width = Pt(1)
txt(sl, "M.Tech CSE  |  Reg No: 2402CUKRM04  |  Roll No: 4",
    3.5, 1.0, 6.3, 0.45, size=11, color=RGBColor(0xA5, 0xB4, 0xFC),
    align=PP_ALIGN.CENTER, bold=True)

# Main title
txt(sl, "SEAL-DSA", 1.5, 1.7, 10.3, 1.5, size=72, bold=True,
    color=C_WHITE, align=PP_ALIGN.CENTER)

multiline_txt(sl, [
    dict(text="Self-Evolving Adaptive Learner for", size=26, color=C_MUTED,
         align=PP_ALIGN.CENTER),
    dict(text="Data Structures & Algorithms", size=26, color=C_CYAN,
         bold=True, align=PP_ALIGN.CENTER),
], 1.5, 3.1, 10.3, 0.9)

divider(sl, 4.2)

multiline_txt(sl, [
    dict(text="Taqaddus Shafi", size=18, bold=True, color=C_WHITE,
         align=PP_ALIGN.CENTER),
    dict(text="Phase 2 Progress Report  •  June 2026", size=15,
         color=C_MUTED, align=PP_ALIGN.CENTER),
], 1.5, 4.35, 10.3, 0.9)

# Badge
b2 = box(sl, 2.8, 5.5, 7.7, 0.55, bg_color=RGBColor(0x1E, 0x29, 0x3B))
b2.line.color.rgb = C_GREEN
b2.line.width = Pt(1)
txt(sl, "🧠  Self-Teaching LLM Fine-Tuning with LoRA + EWC",
    2.8, 5.5, 7.7, 0.55, size=13, color=C_GREEN,
    align=PP_ALIGN.CENTER, bold=True)

footer(sl, 1)


# ════════════════════════════════════════════════════════════
# SLIDE 2 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Project Overview", 0.4, 0.15, 10, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Teaching a language model to become its own DSA tutor",
    0.4, 0.68, 12, 0.35, size=15, color=C_MUTED)
divider(sl, 1.05)

cards = [
    (C_PURPLE, "🎯  Problem Statement",
     [dict(text="LLMs give generic answers to DSA questions. They lack structured,\ncurriculum-aligned teaching ability for progressive skill building.", size=13, color=C_MUTED)]),
    (C_GREEN, "💡  Proposed Solution — SEAL",
     [dict(text="A self-evolving loop where the model generates questions, answers them,\nself-evaluates, and updates its own weights — an autonomous DSA tutor.", size=13, color=C_MUTED)]),
    (C_CYAN, "🔧  Technical Approach",
     [dict(text="• Qwen2.5-1.5B-Instruct as base model\n• 4-bit QLoRA for memory efficiency\n• EWC to prevent catastrophic forgetting\n• IRT-based curriculum scheduling", size=13, color=C_MUTED)]),
    (C_AMBER, "📊  Key Contributions",
     [dict(text="• Novel self-teaching training loop\n• IRT competence-based curriculum\n• Automated forgetting detection\n• Runs on free Google Colab (T4 GPU)", size=13, color=C_MUTED)]),
]
positions = [(0.3, 1.15), (6.85, 1.15), (0.3, 4.0), (6.85, 4.0)]
for (cx, cy), (col, ttl, blines) in zip(positions, cards):
    card(sl, cx, cy, 6.3, 2.7, accent_color=col, title=ttl, body_lines=blines)

footer(sl, 2)


# ════════════════════════════════════════════════════════════
# SLIDE 3 — ARCHITECTURE
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "System Architecture", 0.4, 0.15, 10, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "The SEAL self-teaching training loop",
    0.4, 0.68, 12, 0.35, size=15, color=C_MUTED)
divider(sl, 1.05)

# Flow boxes
flow = [
    (C_CYAN,   "📚", "Curriculum\nScheduler",   "IRT-based"),
    (C_PURPLE, "❓", "Question\nGenerator",      "Topic-aware"),
    (C_PURPLE, "💬", "Answer\nGenerator",        "Chat template"),
    (C_PURPLE, "✅", "Self-Evaluator",           "Rubric+code"),
    (C_GREEN,  "⚡", "LoRA Updater",            "Weight update"),
]
xs = [0.25, 2.75, 5.25, 7.75, 10.25]
for i, ((col, icon, label, detail), x) in enumerate(zip(flow, xs)):
    b = box(sl, x, 1.25, 2.3, 1.5, bg_color=C_BG2)
    b.line.color.rgb = col
    b.line.width = Pt(1.5)
    txt(sl, icon,   x+0.9, 1.3,  0.6, 0.4, size=20, align=PP_ALIGN.CENTER)
    txt(sl, label,  x+0.05, 1.7, 2.2, 0.5, size=12, bold=True,
        color=C_LIGHT, align=PP_ALIGN.CENTER)
    txt(sl, detail, x+0.05, 2.2, 2.2, 0.4, size=10, color=C_MUTED,
        align=PP_ALIGN.CENTER)
    if i < 4:
        txt(sl, "→", x+2.35, 1.8, 0.4, 0.4, size=20, color=C_MUTED,
            align=PP_ALIGN.CENTER)

# Loop back arrow label
txt(sl, "↻  Loop repeats each epoch", 5.0, 2.9, 3.3, 0.4,
    size=13, color=C_MUTED, align=PP_ALIGN.CENTER, bold=True)

# Support systems
support = [
    (C_GREEN,  "🛡️  EWC Regularization", "Fisher Information Matrix\nDynamic λ adaptation"),
    (C_GREEN,  "🔍  Forgetting Detector", "Cross-topic monitoring\nAuto-flags at-risk topics"),
    (C_GREEN,  "💾  Checkpoint Manager",  "Auto-saves to Drive\nResume from any epoch"),
]
sxs = [0.7, 4.7, 8.7]
for (col, ttl, det), sx in zip(support, sxs):
    b = box(sl, sx, 3.6, 3.6, 1.65, bg_color=C_BG2)
    b.line.color.rgb = col
    b.line.width = Pt(1)
    txt(sl, ttl, sx+0.15, 3.7, 3.35, 0.45, size=13, bold=True, color=C_GREEN)
    txt(sl, det, sx+0.15, 4.15, 3.35, 0.9, size=11, color=C_MUTED)

txt(sl, "Support Systems (run in background every epoch)",
    0.4, 3.4, 12.5, 0.3, size=12, color=C_MUTED, italic=True)

footer(sl, 3)


# ════════════════════════════════════════════════════════════
# SLIDE 4 — 17 CRITICAL BUG FIXES
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Phase 2: 17 Critical Bug Fixes", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Resolved all pipeline failures — LoRA fine-tuning is now fully functional",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

# Table header
box(sl, 0.3, 1.12, 12.73, 0.42, bg_color=RGBColor(0x1E, 0x1E, 0x4E))
hdrs = [("#", 0.35, 0.5), ("Bug", 0.9, 2.5), ("Impact", 3.5, 4.0),
        ("File", 7.6, 2.5), ("Status", 10.2, 2.7)]
for ht, hx, hw in hdrs:
    txt(sl, ht, hx, 1.15, hw, 0.38, size=11, bold=True,
        color=RGBColor(0xA5, 0xB4, 0xFC), align=PP_ALIGN.LEFT)

rows = [
    ("1", "Config inheritance missing",   "colab_optimized.yaml ignored defaults",  "config.py",          True),
    ("2", "AMP + QLoRA conflict",         "NaN gradients — training crash",          "parameter_updater",  True),
    ("3", "Label masking misalignment",   "Wrong token boundaries → zero learning",  "parameter_updater",  True),
    ("4", "No chat template (5 files)",   "Train/inference format mismatch",         "5 files",            True),
    ("5", "EWC Fisher in eval mode",      "Incorrect regularization gradients",      "ewc.py",             True),
    ("6", "grad_norm uninitialized",      "UnboundLocalError crash in loop",         "parameter_updater",  True),
    ("7", "Forgetting report keys missing","Dynamic EWC λ adaptation broken",        "forgetting_detector",True),
    ("8", "Code parsed after lowercase",  "Valid Python rejected as syntax error",   "forgetting_detector",True),
]
row_colors = [C_BG, C_BG2]
for i, (num, bug, impact, file, fixed) in enumerate(rows):
    ry = 1.57 + i * 0.59
    box(sl, 0.3, ry, 12.73, 0.57, bg_color=row_colors[i % 2])
    txt(sl, num,    0.38, ry+0.07, 0.45, 0.42, size=11, color=C_MUTED)
    txt(sl, bug,    0.88, ry+0.07, 2.5,  0.42, size=11, bold=True, color=C_LIGHT)
    impact_col = C_RED if "crash" in impact.lower() or "zero" in impact.lower() or "NaN" in impact else C_MUTED
    txt(sl, impact, 3.45, ry+0.07, 4.0,  0.42, size=11, color=impact_col)
    txt(sl, file,   7.55, ry+0.07, 2.5,  0.42, size=10, color=C_MUTED, italic=True)
    txt(sl, "✅ Fixed", 10.2, ry+0.07, 2.7, 0.42, size=11, bold=True, color=C_GREEN)

txt(sl, "+ 9 additional fixes: redundant forward pass, deprecated APIs, GPU typos, negative training, and more",
    0.4, 6.9, 12.5, 0.4, size=11, color=C_MUTED, italic=True)

footer(sl, 4)


# ════════════════════════════════════════════════════════════
# SLIDE 5 — KEY FIXES DEEP DIVE
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Key Technical Fixes", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "The three critical issues that prevented all LoRA learning",
    0.4, 0.68, 12, 0.35, size=15, color=C_MUTED)
divider(sl, 1.05)

fixes = [
    ("Fix 1 — AMP + Quantization Conflict",
     "BEFORE: GradScaler + float16 autocast on 4-bit NF4 model → NaN gradients, training crash",
     "AFTER:  AMP auto-disabled when quantization_enabled=True → clean stable gradients"),
    ("Fix 2 — Prompt Format Mismatch",
     'BEFORE: prompt = "Question: ... Answer:" (raw text, model never saw this during pretraining)',
     "AFTER:  tokenizer.apply_chat_template() — training format exactly matches Qwen2.5-Instruct"),
    ("Fix 3 — Label Masking (Token Boundary)",
     "BEFORE: Tokenized full text together, guessed split using character count → off by 5-10 tokens",
     "AFTER:  Tokenize question & answer separately → mask exactly len(input_ids) tokens"),
]

for i, (title, before, after) in enumerate(fixes):
    fy = 1.15 + i * 1.9
    b = box(sl, 0.3, fy, 12.73, 1.75, bg_color=C_BG2)
    b.line.color.rgb = C_RED
    b.line.width = Pt(1)
    txt(sl, title, 0.5, fy+0.08, 12.4, 0.38, size=14, bold=True, color=C_LIGHT)
    box(sl, 0.4, fy+0.48, 5.9, 0.55, bg_color=RGBColor(0x3B, 0x1A, 0x1A))
    txt(sl, "🔴 " + before, 0.5, fy+0.49, 5.8, 0.52, size=10.5, color=RGBColor(0xFC, 0xA5, 0xA5))
    box(sl, 6.55, fy+0.48, 6.4, 0.55, bg_color=RGBColor(0x0F, 0x2E, 0x20))
    txt(sl, "🟢 " + after, 6.65, fy+0.49, 6.25, 0.52, size=10.5, color=RGBColor(0x6E, 0xE7, 0xB7))
    txt(sl, "→", 6.25, fy+0.55, 0.4, 0.4, size=16, color=C_AMBER, align=PP_ALIGN.CENTER)

footer(sl, 5)


# ════════════════════════════════════════════════════════════
# SLIDE 6 — TRAINING RESULTS
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Training Results", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Measured on Google Colab T4 GPU — 4-bit quantization",
    0.4, 0.68, 12, 0.35, size=15, color=C_MUTED)
divider(sl, 1.05)

# Metric boxes
metrics = [
    ("112",  "LoRA Layers\nUpdated",     C_GREEN),
    ("5.5×", "Weight Change\nGrowth",    C_CYAN),
    ("0.0071","Avg Delta\nNorm",         C_PURPLE),
    ("~5 GB", "Peak GPU\nMemory",        C_AMBER),
]
mxs = [0.35, 3.55, 6.75, 9.95]
for (val, lbl, col), mx in zip(metrics, mxs):
    b = box(sl, mx, 1.15, 3.0, 1.65, bg_color=C_BG2)
    b.line.color.rgb = col
    b.line.width = Pt(2)
    txt(sl, val, mx, 1.25, 3.0, 0.85, size=38, bold=True, color=col,
        align=PP_ALIGN.CENTER)
    txt(sl, lbl, mx, 2.05, 3.0, 0.65, size=12, color=C_MUTED,
        align=PP_ALIGN.CENTER)

txt(sl, "LoRA Weight Evolution Across Epochs", 0.4, 2.95, 12, 0.4,
    size=16, bold=True, color=C_LIGHT)

box(sl, 0.3, 3.37, 12.73, 0.42, bg_color=RGBColor(0x1E, 0x1E, 0x4E))
for htxt, hx, hw in [("Metric",0.4,4.0),("Epoch 1",4.5,2.5),
                      ("Epoch 2",7.1,2.5),("Change",9.8,3.0)]:
    txt(sl, htxt, hx, 3.4, hw, 0.38, size=11, bold=True,
        color=RGBColor(0xA5, 0xB4, 0xFC))

table_rows = [
    ("Delta Norms (avg)", "0.0013", "0.0071", "+446%  ↑"),
    ("Delta Norms (max)", "0.0018", "0.0108", "+500%  ↑"),
    ("Delta Norms (min)", "0.0007", "0.0039", "+457%  ↑"),
    ("Layers Merged",     "112",    "112",     "Full coverage"),
    ("All Answers Different","✅ 3/3","✅ 3/3","Model is learning"),
]
rcolors = [C_BG, C_BG2]
for i, (metric, e1, e2, change) in enumerate(table_rows):
    ry = 3.82 + i * 0.58
    box(sl, 0.3, ry, 12.73, 0.56, bg_color=rcolors[i % 2])
    txt(sl, metric, 0.45, ry+0.1, 3.9, 0.42, size=12, color=C_LIGHT, bold=True)
    txt(sl, e1, 4.55, ry+0.1, 2.4, 0.42, size=12, color=C_MUTED)
    txt(sl, e2, 7.15, ry+0.1, 2.4, 0.42, size=12, bold=True, color=C_GREEN)
    txt(sl, change, 9.85, ry+0.1, 2.9, 0.42, size=12, bold=True, color=C_GREEN)

footer(sl, 6)


# ════════════════════════════════════════════════════════════
# SLIDE 7 — QUALITATIVE COMPARISON
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Base Model vs SEAL Model", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Side-by-side answer quality on DSA questions after 2 epochs of SEAL training",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

comparisons = [
    ("Q1: Implement Kadane's algorithm",
     "Jumps straight to code. Generic variable names without explanation.",
     "Explains max_current and max_global with bullet-point definitions BEFORE code. Teaching-oriented."),
    ("Q2: Explain BFS vs DFS traversal",
     '"Here\'s how they work" — lists facts separately. No comparison.',
     '"They approach the task differently" — comparative framing. Adds Pros section with practical insight.'),
    ("Q3: What is dynamic programming?",
     'Generic "cache" terminology. Lists concepts as disconnected bullets.',
     '"Avoids redundant calculations, reduces time complexity vs naive solutions" — specific & comparative.'),
]
for i, (q, before, after) in enumerate(comparisons):
    fy = 1.12 + i * 2.05
    txt(sl, q, 0.35, fy, 12.5, 0.32, size=12, bold=True, color=C_AMBER, italic=True)
    b1 = box(sl, 0.3, fy+0.33, 6.2, 1.55, bg_color=RGBColor(0x1F, 0x12, 0x12))
    b1.line.color.rgb = C_RED; b1.line.width = Pt(1)
    txt(sl, "📘 BASE MODEL", 0.4, fy+0.35, 3, 0.28, size=10, bold=True,
        color=RGBColor(0xFC, 0xA5, 0xA5))
    txt(sl, before, 0.45, fy+0.62, 6.0, 1.1, size=11, color=RGBColor(0xFC, 0xA5, 0xA5), wrap=True)
    b2 = box(sl, 6.7, fy+0.33, 6.3, 1.55, bg_color=RGBColor(0x0A, 0x1F, 0x16))
    b2.line.color.rgb = C_GREEN; b2.line.width = Pt(1)
    txt(sl, "📗 SEAL MODEL", 6.8, fy+0.35, 3.5, 0.28, size=10, bold=True, color=C_GREEN)
    txt(sl, after, 6.85, fy+0.62, 6.05, 1.1, size=11, color=RGBColor(0x6E, 0xE7, 0xB7), wrap=True)

footer(sl, 7)


# ════════════════════════════════════════════════════════════
# SLIDE 8 — IMPROVEMENT PATTERNS
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Observed Improvement Patterns", 0.4, 0.15, 12, 0.55, size=30, bold=True, color=C_WHITE)
txt(sl, "How the SEAL-trained model differs from the base model",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

patterns = [
    (C_GREEN,  "📝 More Structured",
     "Provides explain→code format instead of jumping to implementation.\nAdds numbered steps before algorithms."),
    (C_CYAN,   "⚖️ Comparative Framing",
     '"They approach the task differently" — frames concepts as contrasts,\nmirroring how DSA is taught in classrooms.'),
    (C_PURPLE, "🎓 Teaching-Oriented",
     "Explains variable purposes before code. Adds Pros/Cons sections.\nBehaves more like a tutor than a code generator."),
    (C_AMBER,  "📊 Complexity-Aware",
     "Consistently mentions time/space complexity.\nExplicitly references optimal substructure & overlapping subproblems."),
]
pxs = [(0.3, 1.15), (6.85, 1.15), (0.3, 3.65), (6.85, 3.65)]
for (px, py), (col, ttl, body) in zip(pxs, patterns):
    b = box(sl, px, py, 6.3, 2.3, bg_color=C_BG2)
    b.line.color.rgb = col; b.line.width = Pt(1.5)
    txt(sl, ttl,  px+0.15, py+0.12, 6.0, 0.38, size=14, bold=True, color=C_LIGHT)
    txt(sl, body, px+0.15, py+0.55, 6.0, 1.55, size=12, color=C_MUTED, wrap=True)

# Summary row
box(sl, 0.3, 6.0, 12.73, 1.3, bg_color=C_BG2)
summary = [("Answer Style", "Code-first, generic", "Explain-first, structured"),
           ("Comparisons",  "Lists facts separately","Comparative framing"),
           ("Complexity",   "Sometimes","Consistently included")]
for i, (dim, bef, aft) in enumerate(summary):
    sx = 0.45 + i * 4.25
    txt(sl, dim, sx, 6.05, 4.0, 0.3, size=11, bold=True, color=C_LIGHT)
    txt(sl, "✗ " + bef, sx, 6.38, 4.0, 0.28, size=10, color=C_RED)
    txt(sl, "✓ " + aft, sx, 6.68, 4.0, 0.28, size=10, color=C_GREEN)

footer(sl, 8)


# ════════════════════════════════════════════════════════════
# SLIDE 9 — PHASE 1 vs PHASE 2 COMPARISON TABLE
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Phase 1 → Phase 2: Before & After", 0.4, 0.15, 12, 0.55,
    size=30, bold=True, color=C_WHITE)
txt(sl, "A clear picture of progress made across all dimensions",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

box(sl, 0.3, 1.12, 12.73, 0.42, bg_color=RGBColor(0x1E, 0x1E, 0x4E))
for ht, hx, hw in [("Aspect",0.4,4.0),("Phase 1 — Before",4.5,4.1),("Phase 2 — After",8.75,4.1)]:
    txt(sl, ht, hx, 1.15, hw, 0.38, size=12, bold=True,
        color=RGBColor(0xA5, 0xB4, 0xFC))

compare_rows = [
    ("Training Status",     "❌ LoRA weights not updating",     "✅ 112 layers updating/epoch"),
    ("Gradient Flow",       "❌ NaN / zero gradients",          "✅ Stable (norm 0.3–2.0)"),
    ("Config System",       "❌ Inheritance broken",            "✅ Deep-merge _base_ working"),
    ("Prompt Format",       "❌ Raw text (mismatched)",         "✅ Chat template aligned"),
    ("Label Masking",       "❌ Character-level (wrong)",       "✅ Token-level (exact)"),
    ("AMP + Quantization",  "❌ Conflicting (NaN)",             "✅ AMP disabled for QLoRA"),
    ("EWC Regularization",  "❌ Eval mode (wrong gradients)",   "✅ Train mode (correct)"),
    ("Model Comparison",    "⚠️  SAME (no difference)",        "✅ DIFFERENT (3/3 questions)"),
    ("LoRA Delta Norms",    "0.0000 (no learning)",             "0.0071 avg (active learning)"),
    ("Syntax Validation",   "Unchecked",                        "✅ All 24 files pass"),
]
rcolors = [C_BG, C_BG2]
for i, (aspect, before, after) in enumerate(compare_rows):
    ry = 1.57 + i * 0.535
    box(sl, 0.3, ry, 12.73, 0.52, bg_color=rcolors[i % 2])
    txt(sl, aspect, 0.42, ry+0.08, 3.9, 0.42, size=11, bold=True, color=C_LIGHT)
    bcol = C_RED if "❌" in before or "⚠️" in before else C_MUTED
    txt(sl, before, 4.52, ry+0.08, 4.0, 0.42, size=11, color=bcol)
    txt(sl, after,  8.77, ry+0.08, 4.0, 0.42, size=11, bold=True, color=C_GREEN)

footer(sl, 9)


# ════════════════════════════════════════════════════════════
# SLIDE 10 — NOVEL CONTRIBUTIONS
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Novel Research Contributions", 0.4, 0.15, 12, 0.55, size=30, bold=True, color=C_WHITE)
txt(sl, "Unique innovations in the SEAL-DSA framework",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

b = box(sl, 0.3, 1.15, 12.73, 2.1, bg_color=C_BG2)
b.line.color.rgb = C_PURPLE; b.line.width = Pt(1.5)
txt(sl, "🔄", 0.5, 1.25, 0.8, 0.8, size=36)
txt(sl, "Self-Teaching Loop", 1.4, 1.2, 11.2, 0.45, size=16, bold=True, color=C_LIGHT)
txt(sl, "Unlike traditional fine-tuning which requires external datasets, SEAL generates its own training data through a closed loop:\nQuestion Generation → Self-Answering → Self-Evaluation → Weight Update\nThe model reinforces correct answers and discards incorrect ones — fully autonomous.",
    1.4, 1.65, 11.2, 1.4, size=12, color=C_MUTED, wrap=True)

b2 = box(sl, 0.3, 3.45, 6.2, 2.4, bg_color=C_BG2)
b2.line.color.rgb = C_GREEN; b2.line.width = Pt(1.5)
txt(sl, "📈  IRT-Based Competence Estimation", 0.5, 3.52, 5.8, 0.38, size=14, bold=True, color=C_GREEN)
txt(sl, "Applies Item Response Theory (1PL model) from psychometrics.\nEstimates model ability (θ) per topic.\nPrioritizes Zone of Proximal Development (θ ∈ [-1, 1]).\n\nFormula: P(correct|θ,b) = 1/(1+exp(-(θ-b)))",
    0.5, 3.95, 5.8, 1.75, size=12, color=C_MUTED, wrap=True)

b3 = box(sl, 6.7, 3.45, 6.3, 2.4, bg_color=C_BG2)
b3.line.color.rgb = C_CYAN; b3.line.width = Pt(1.5)
txt(sl, "🛡️  Dynamic EWC + Forgetting Detection", 6.9, 3.52, 5.9, 0.38, size=14, bold=True, color=C_CYAN)
txt(sl, "Combines Elastic Weight Consolidation with real-time\nforgetting detection across all learned topics.\nLambda (λ) auto-adapts based on measured forgetting rates.\nPrevents catastrophic forgetting while enabling new learning.",
    6.9, 3.95, 5.9, 1.75, size=12, color=C_MUTED, wrap=True)

footer(sl, 10)


# ════════════════════════════════════════════════════════════
# SLIDE 11 — TECHNICAL STACK
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Technical Stack", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Hardware-optimized for Google Colab Free Tier — T4 GPU",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

stack = [
    (C_PURPLE, "🤖  Base Model",
     "Qwen2.5-1.5B-Instruct\n1.5 billion parameters\n4-bit NF4 quantization\n~5GB GPU memory"),
    (C_GREEN,  "⚡  LoRA Config",
     "Rank: 4  |  Alpha: 8\nTarget: q/k/v/o projections\nTrainable params: ~1.7M\nOnly 0.11% of weights"),
    (C_CYAN,   "🏗️  Training Config",
     "Batch size: 2\nGrad accumulation: 8 steps\nEffective batch: 16\nLearning rate: 2×10⁻⁴"),
    (C_AMBER,  "🛡️  EWC Config",
     "Lambda (λ): 0.3\nFisher samples: 100\nDynamic λ adaptation\nPrevents forgetting"),
    (C_PURPLE, "📚  Curriculum",
     "7 core DSA topics\n16-week progressive schedule\nIRT competence scoring\nZone of Proximal Dev."),
    (C_GREEN,  "💻  Hardware",
     "Google Colab Free Tier\nNVIDIA T4 (15GB VRAM)\n~30-60 min per epoch\nAuto-save to Drive"),
]
positions = [(0.3,1.15),(4.5,1.15),(8.7,1.15),(0.3,4.0),(4.5,4.0),(8.7,4.0)]
for (sx, sy), (col, ttl, body) in zip(positions, stack):
    b = box(sl, sx, sy, 3.9, 2.7, bg_color=C_BG2)
    b.line.color.rgb = col; b.line.width = Pt(1.5)
    txt(sl, ttl,  sx+0.15, sy+0.1, 3.7, 0.38, size=13, bold=True, color=C_LIGHT)
    txt(sl, body, sx+0.15, sy+0.55, 3.7, 2.0, size=12, color=C_MUTED, wrap=True)

footer(sl, 11)


# ════════════════════════════════════════════════════════════
# SLIDE 12 — PROJECT TIMELINE
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Project Timeline", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Phase 1 → Phase 2 progression and milestones",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

# Phase 1
box(sl, 0.3, 1.15, 6.2, 0.42, bg_color=RGBColor(0x1E, 0x1E, 0x4E))
txt(sl, "Phase 1 — Foundation", 0.45, 1.18, 5.9, 0.38, size=14, bold=True,
    color=C_MUTED)

p1_items = [
    ("Architecture Design",   "Designed modular SEAL pipeline with 6 core modules"),
    ("Codebase Implementation","Built 24 Python files, 4,000+ lines of code"),
    ("DSA Curriculum",        "Defined 7 topics, 70+ subtopics, difficulty progression"),
    ("Initial Training",      "Identified that LoRA weights were NOT updating — pipeline broken"),
]
for i, (title, detail) in enumerate(p1_items):
    ty = 1.68 + i * 1.2
    box(sl, 0.5, ty, 0.2, 0.2, bg_color=C_MUTED)  # dot
    box(sl, 0.6, ty+0.1, 5.6, 0.02, bg_color=C_MUTED)  # line down
    txt(sl, title,  0.5, ty, 5.7, 0.35, size=13, bold=True, color=C_LIGHT)
    txt(sl, detail, 0.6, ty+0.35, 5.6, 0.55, size=11, color=C_MUTED, wrap=True)

# Phase 2
box(sl, 6.85, 1.15, 6.15, 0.42, bg_color=RGBColor(0x0F, 0x2E, 0x20))
txt(sl, "Phase 2 — Debugging & Validation", 7.0, 1.18, 5.9, 0.38, size=14, bold=True,
    color=C_GREEN)

p2_items = [
    ("Root Cause Analysis",  "Identified 17 critical bugs across 10 source files"),
    ("Pipeline Fixes",       "Fixed AMP, label masking, chat templates, EWC, configs"),
    ("Training Validation",  "Verified 112 LoRA layers updating — delta norms measurable"),
    ("Model Comparison",     "Confirmed qualitative improvement in DSA teaching quality"),
]
for i, (title, detail) in enumerate(p2_items):
    ty = 1.68 + i * 1.2
    box(sl, 7.05, ty, 0.2, 0.2, bg_color=C_GREEN)
    txt(sl, title,  7.05, ty, 5.7, 0.35, size=13, bold=True, color=C_GREEN)
    txt(sl, detail, 7.15, ty+0.35, 5.6, 0.55, size=11, color=C_MUTED, wrap=True)

footer(sl, 12)


# ════════════════════════════════════════════════════════════
# SLIDE 13 — FUTURE WORK
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl); accent_bar(sl)

txt(sl, "Future Work", 0.4, 0.15, 12, 0.55, size=32, bold=True, color=C_WHITE)
txt(sl, "Planned improvements for continued development",
    0.4, 0.68, 12, 0.35, size=14, color=C_MUTED)
divider(sl, 1.05)

future = [
    (C_PURPLE, "🔁  Extended Training (5+ Epochs)",
     "Current results show 2 epochs of training. Running for 5+ epochs across the full 7-topic curriculum will produce stronger model specialization, larger LoRA delta norms, and measurable score improvements on automated benchmarks."),
    (C_GREEN,  "📊  Quantitative Evaluation",
     "Implement automated scoring metrics: code execution accuracy, perplexity reduction, and rubric-based quality scores across all 7 DSA topics with statistical significance testing to formally validate improvement."),
    (C_CYAN,   "⚔️  DPO Contrastive Training",
     "Activate the implemented Direct Preference Optimization (DPO) loss function to further improve answer quality by contrasting correct answers against incorrect ones during the training loop."),
    (C_AMBER,  "👨‍🎓  User Study",
     "Conduct a human evaluation with Computer Science students comparing SEAL-DSA responses vs base model responses for educational quality, clarity, correctness, and teaching effectiveness."),
]
fxs = [(0.3, 1.15), (6.85, 1.15), (0.3, 4.0), (6.85, 4.0)]
for (fx, fy), (col, ttl, body) in zip(fxs, future):
    b = box(sl, fx, fy, 6.3, 2.7, bg_color=C_BG2)
    b.line.color.rgb = col; b.line.width = Pt(1.5)
    txt(sl, ttl,  fx+0.15, fy+0.1,  6.0, 0.38, size=13, bold=True, color=C_LIGHT)
    txt(sl, body, fx+0.15, fy+0.55, 6.0, 2.0,  size=12, color=C_MUTED, wrap=True)

footer(sl, 13)


# ════════════════════════════════════════════════════════════
# SLIDE 14 — CONCLUSION / THANK YOU
# ════════════════════════════════════════════════════════════
sl = add_slide()
bg(sl)
box(sl, 0, 0, 13.33, 7.5, bg_color=RGBColor(0x0A, 0x0E, 0x1F))
accent_bar(sl)

txt(sl, "Key Takeaways", 1.5, 0.5, 10.3, 0.65, size=36, bold=True,
    color=C_WHITE, align=PP_ALIGN.CENTER)

takeaways = [
    ("✅", "17 critical bugs fixed across 10 source files — pipeline fully functional"),
    ("✅", "LoRA training validated — 112 layers updating, 5.5× weight growth per epoch"),
    ("✅", "Qualitative improvement confirmed — more structured, teaching-oriented DSA answers"),
    ("✅", "All 24 Python files pass syntax validation with zero circular imports"),
    ("✅", "Runs on free hardware — optimized for Google Colab T4 GPU (5GB VRAM)"),
    ("🚀", "SEAL-DSA is learning — the self-teaching loop works"),
]
for i, (icon, text) in enumerate(takeaways):
    ty = 1.3 + i * 0.72
    b = box(sl, 1.5, ty, 10.3, 0.62, bg_color=C_BG2)
    b.line.color.rgb = C_GREEN if icon == "✅" else C_PURPLE
    b.line.width = Pt(1)
    txt(sl, icon, 1.65, ty+0.1, 0.5, 0.42, size=14, align=PP_ALIGN.CENTER)
    txt(sl, text, 2.3, ty+0.1, 9.3, 0.42, size=13, bold=(icon=="🚀"),
        color=C_LIGHT if icon=="✅" else C_PURPLE)

divider(sl, 5.72)
multiline_txt(sl, [
    dict(text="Taqaddus Shafi", size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER),
    dict(text="M.Tech CSE  |  Reg No: 2402CUKRM04  |  Roll No: 4", size=13, color=C_MUTED, align=PP_ALIGN.CENTER, space_before=4),
    dict(text="github.com/Taqaddusshafi/dsaseal", size=13, color=C_CYAN, align=PP_ALIGN.CENTER, space_before=4),
], 1.5, 5.82, 10.3, 1.2)

footer(sl, 14)


# ── Save ────────────────────────────────────────────────────
import os
os.makedirs("presentation", exist_ok=True)
out = "presentation/SEAL_DSA_Phase2.pptx"
prs.save(out)
print(f"✅ Saved: {out}")
