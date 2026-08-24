from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "COINS_论文精读摘要.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FONT = "MSYH"
FONT_BOLD = "MSYH-Bold"
pdfmetrics.registerFont(TTFont(FONT, r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F95")
TEAL = colors.HexColor("#4D9A9A")
LIGHT_BLUE = colors.HexColor("#EAF3F7")
LIGHT_TEAL = colors.HexColor("#EAF6F3")
LIGHT_GRAY = colors.HexColor("#F4F6F8")
MID_GRAY = colors.HexColor("#6B7280")
DARK = colors.HexColor("#1F2933")


def p(text, style):
    return Paragraph(text, style)


def bullet(text, style):
    return Paragraph(f"<bullet>&bull;</bullet>{text}", style)


class MethodDiagram(Flowable):
    def __init__(self, width=165 * mm, height=52 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw_box(self, c, x, y, w, h, title, lines, fill):
        c.setFillColor(fill)
        c.setStrokeColor(colors.HexColor("#9AA9B5"))
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont(FONT_BOLD, 9)
        c.drawCentredString(x + w / 2, y + h - 15, title)
        c.setFillColor(DARK)
        c.setFont(FONT, 7.5)
        yy = y + h - 29
        for line in lines:
            c.drawCentredString(x + w / 2, yy, line)
            yy -= 11

    def arrow(self, c, x1, y1, x2, y2):
        c.setStrokeColor(BLUE)
        c.setLineWidth(1.2)
        c.line(x1, y1, x2, y2)
        c.line(x2, y2, x2 - 5, y2 + 3)
        c.line(x2, y2, x2 - 5, y2 - 3)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        c.setFillColor(LIGHT_GRAY)
        c.roundRect(0, 0, w, h, 8, fill=1, stroke=0)
        bw, bh = 43 * mm, 26 * mm
        y = h - bh - 11
        self.draw_box(c, 7 * mm, y, bw, bh, "输入", ["Item ID + 内容特征", "用户画像 + 查询上下文"], colors.white)
        self.draw_box(c, 61 * mm, y, bw, bh, "RQ 共享信息", ["RQ1 / RQ2 / RQ3", "语义与协同共性"], LIGHT_BLUE)
        self.draw_box(c, 115 * mm, y, bw, bh, "OPQ 个性信息", ["OPQ1 / OPQ2", "细粒度差异"], LIGHT_TEAL)
        self.arrow(c, 50 * mm, y + bh / 2, 60 * mm, y + bh / 2)
        self.arrow(c, 104 * mm, y + bh / 2, 114 * mm, y + bh / 2)
        gy = 8 * mm
        gw, gh = 78 * mm, 13 * mm
        self.draw_box(c, 7 * mm, gy, gw, gh, "Adaptive Gate", ["Tg = DNN(context) -> 动态选择 ID / RQ"], colors.HexColor("#FFF4DB"))
        self.draw_box(c, 91 * mm, gy, 67 * mm, gh, "CTR 表示", ["Ic + lambda I_OPQ -> BCE"], colors.HexColor("#F0EAF7"))
        self.arrow(c, 82 * mm, gy + gh / 2, 90 * mm, gy + gh / 2)
        c.setFillColor(MID_GRAY)
        c.setFont(FONT, 7)
        c.drawString(8 * mm, 2 * mm, "训练目标：BCE + alpha1 Ltrans + alpha2 Lcont")


class CoinsDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="coins", frames=frame, onPage=self.decorate)])

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6DEE5"))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, A4[1] - 17 * mm, A4[0] - doc.rightMargin, A4[1] - 17 * mm)
        canvas.setFont(FONT_BOLD, 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(doc.leftMargin, A4[1] - 13 * mm, "COINS | 语义 ID 增强的冷启动物品表示")
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MID_GRAY)
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"第 {doc.page} 页")
        canvas.drawString(doc.leftMargin, 10 * mm, "基于 WWW 2026 论文正文整理 | DOI: 10.1145/3774904.3792902")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCN", fontName=FONT_BOLD, fontSize=22, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8 * mm))
styles.add(ParagraphStyle(name="Subtitle", fontName=FONT, fontSize=10, leading=16, textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=7 * mm))
styles.add(ParagraphStyle(name="H1CN", fontName=FONT_BOLD, fontSize=15, leading=21, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="H2CN", fontName=FONT_BOLD, fontSize=11, leading=16, textColor=BLUE, spaceBefore=2.5 * mm, spaceAfter=1.5 * mm))
styles.add(ParagraphStyle(name="BodyCN", fontName=FONT, fontSize=9.1, leading=15, textColor=DARK, alignment=TA_LEFT, spaceAfter=2.2 * mm))
styles.add(ParagraphStyle(name="BulletCN", fontName=FONT, fontSize=8.8, leading=14, leftIndent=5 * mm, firstLineIndent=-3 * mm, textColor=DARK, spaceAfter=1.3 * mm))
styles.add(ParagraphStyle(name="SmallCN", fontName=FONT, fontSize=7.6, leading=11, textColor=MID_GRAY, spaceAfter=1.5 * mm))
styles.add(ParagraphStyle(name="Callout", fontName=FONT_BOLD, fontSize=10, leading=16, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=colors.HexColor("#B9D5E3"), borderWidth=0.6, borderPadding=7, spaceBefore=2 * mm, spaceAfter=4 * mm))
styles.add(ParagraphStyle(name="TableCN", fontName=FONT, fontSize=7.5, leading=10, textColor=DARK, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableHeadCN", fontName=FONT_BOLD, fontSize=7.6, leading=10, textColor=colors.white, alignment=TA_CENTER))


def table(data, widths, header_rows=1, font_size=7.5):
    t = Table(data, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D1D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(header_rows, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    t.setStyle(TableStyle(commands))
    return t


story = []
story.append(Spacer(1, 14 * mm))
story.append(p("COINS 论文精读摘要", styles["TitleCN"]))
story.append(p("Semantic Ids Enhanced Cold Item Representation for Click-through Rate Prediction in E-commerce Search", styles["Subtitle"]))
story.append(p("WWW 2026 | Qihang Zhao 等 | Kuaishou Technology", styles["Subtitle"]))
story.append(HRFlowable(width="85%", thickness=1.2, color=TEAL, spaceBefore=3 * mm, spaceAfter=7 * mm))
story.append(p("一句话结论：COINS 将冷启动物品表示拆成“可迁移的共享信息”和“不可被抹平的个体差异”，分别由 RQ 和 OPQ 建模，再用自适应门控决定当前请求更应该依赖 Item ID 还是语义表示。", styles["Callout"]))
story.append(p("本摘要面向深度学习推荐方向研一学生，重点保留方法直觉、公式含义、实验可信度和课程推荐冷启动的迁移价值。", styles["BodyCN"]))
story.append(p("核心贡献", styles["H1CN"]))
for text in [
    "RQ 编码：迁移物品之间共享的语义与协同信息。",
    "OPQ 编码：通过对比学习保留每个物品的细粒度差异。",
    "Adaptive Gate：根据用户、查询和物品上下文，动态调整 Item ID 与 RQ 的信息比例。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("实验亮点", styles["H1CN"]))
story.append(p("在真实电商搜索日志上，COINS 的冷物品 AUC 为 0.8528、GAUC 为 0.6301；相较 SaviorRec 分别提升 0.93 和 0.52 个百分点。线上 14 天 A/B 测试中，冷启动订单量提升 9.639%。", styles["BodyCN"]))
story.append(PageBreak())

story.append(p("1. 背景与问题", styles["H1CN"]))
story.append(p("电商平台中新商品持续增加，但新商品缺少点击、购买和转化等协同信号。论文统计显示，新商品约占平台月度更新商品的 30%，超过 60% 的冷启动商品首周点击数少于 5 次。传统 CTR 模型高度依赖 Item ID embedding 和行为统计，因此容易把曝光和流量集中到已有热门商品，形成 Matthew Effect。", styles["BodyCN"]))
story.append(p("论文将冷启动 CTR 形式化为：给定用户画像 U_p、查询 Q、历史行为 H_U、交叉特征 C、物品 ID 和多维物品特征 X，学习模型 f 输出点击概率。训练目标是二元交叉熵 BCE。关键难点不是单纯缺少内容，而是内容信息与协同信息具有不对称性：内容说明物品是什么，协同信息说明用户是否喜欢它。", styles["BodyCN"]))
story.append(p("现有方法的瓶颈", styles["H2CN"]))
for text in [
    "生成式方法可以从热门物品合成冷物品表示，但容易忽略在线分布随时间变化而产生的漂移。",
    "内容-协同对齐方法通常假设两类表示可以直接对齐，容易忽略物品之间的细粒度差异。",
    "仅使用 Semantic ID 可能让同类物品变得过于相似，导致推荐同质化。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("作者与署名单位", styles["H1CN"]))
story.append(p("作者为 Qihang Zhao、Zhongbo Sun、Xiaoyang Zheng、Xian Guo、Siyuan Wang、Zihan Liang、Mingcan Peng、Ben Chen（通讯作者）和 Chenyi Lei。论文署名单位均为 Kuaishou Technology，覆盖杭州和北京。论文未提供作者的院系、职称和教育经历，本文不对其身份作无依据推断。", styles["BodyCN"]))
story.append(p("论文定位", styles["H1CN"]))
story.append(p("这是 WWW 2026 的 4 页工业短文，重点是模型设计和线上验证；相较完整研究论文，训练细节、理论分析和跨平台泛化实验较少。", styles["BodyCN"]))
story.append(PageBreak())

story.append(p("2. 方法总览：RQ + OPQ + Adaptive Gate", styles["H1CN"]))
story.append(p("每个物品由五层 Semantic ID 表示：I_SID = (RQ1, RQ2, RQ3, OPQ1, OPQ2)。RQ 通过 Residual Quantization 表示逐层细化的共性信息；OPQ 表示经过 RQ 量化后剩余的差异信息。", styles["BodyCN"]))
story.append(MethodDiagram())
story.append(Spacer(1, 3 * mm))
story.append(p("2.1 RQ 的自适应迁移", styles["H2CN"]))
story.append(p("先融合三层 RQ 表示：I_RQ = Fuse(RQ1, RQ2, RQ3)。再由门控网络根据上下文 C、用户画像 U_p 和物品特征 X 生成 T_g：T_g = DNN(C, U_p, X)。最终表示为：I_c = T_g I_id + (1 - T_g) I_RQ。", styles["BodyCN"]))
story.append(p("直觉上，热门物品有充分协同信号，因此保留更多 Item ID；新物品的 ID 几乎没有统计意义，因此更多依赖 RQ。门控网络让迁移方向由当前请求决定，而不是对所有物品使用固定比例。", styles["BodyCN"]))
story.append(p("2.2 KL 迁移损失", styles["H2CN"]))
story.append(p("L_trans = T_g KL(sg(I_id), I_RQ) + (1 - T_g) KL(I_id, sg(I_RQ))。其中 sg 是 stop-gradient。T_g 较大时，主要更新 RQ，使其接近 Item ID；T_g 较小时，主要更新 Item ID，使其吸收 RQ 的语义信息。", styles["BodyCN"]))
story.append(p("2.3 OPQ 个体差异学习", styles["H2CN"]))
story.append(p("作者根据 OPQ codebook 向量相似度，为每个物品选取 Top-10 候选正样本，再用 InfoNCE 让正样本更接近、负样本更远。最终表示为：I_f = I_c + lambda I_OPQ。", styles["BodyCN"]))
story.append(PageBreak())

story.append(p("3. 训练目标与实现要点", styles["H1CN"]))
story.append(p("COINS 使用三个损失项联合训练：", styles["BodyCN"]))
for text in [
    "L_BCE：保证最终 CTR 预测准确。",
    "L_trans：约束 Item ID 与 RQ 之间的信息迁移方向。",
    "L_cont：通过 InfoNCE 强化 OPQ 的个体差异。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("L_total = L_BCE + alpha1 L_trans + alpha2 L_cont", styles["Callout"]))
story.append(p("论文设置 alpha1 = 0.01、alpha2 = 0.05、温度系数 tau = 0.1、OPQ 融合系数 lambda = 0.5。模型最后将增强后的物品表示送入 Concat + DNN 排序结构，与用户、查询、交叉特征共同完成 CTR 预测。", styles["BodyCN"]))
story.append(p("如何理解 RQ 和 OPQ？", styles["H1CN"]))
comparison = [
    [p("部分", styles["TableHeadCN"]), p("主要回答的问题", styles["TableHeadCN"]), p("课程推荐中的类比", styles["TableHeadCN"])],
    [p("RQ", styles["TableCN"]), p("这个物品和哪些物品共享语义或协同模式？", styles["TableCN"]), p("Python 基础、机器学习、深度学习属于相近学习路径", styles["TableCN"])],
    [p("OPQ", styles["TableCN"]), p("这个物品相较同类物品有什么独特之处？", styles["TableCN"]), p("教师风格、难度、项目实践、课程时长和先修要求", styles["TableCN"]),],
    [p("Gate", styles["TableCN"]), p("当前请求应该更相信 ID 还是语义？", styles["TableCN"]), p("成熟课程依赖行为，新课程依赖内容和知识结构", styles["TableCN"])],
]
story.append(table(comparison, [24 * mm, 68 * mm, 73 * mm]))
story.append(p("复现时最关键的工程细节", styles["H1CN"]))
for text in [
    "需要明确 RQ-OPQ codebook 的训练方式、每层大小、量化误差和 Semantic ID 更新周期。",
    "需要严格按时间切分冷物品，避免使用测试期曝光或点击统计造成泄漏。",
    "需要记录门控权重 T_g 随物品年龄、曝光量和点击量的变化，以验证其是否学到预期策略。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(PageBreak())

story.append(p("4. 实验结果与证据", styles["H1CN"]))
story.append(p("数据集包含 90 天训练日志和 1 天测试日志，测试集约 5 亿样本，其中约 4 亿为冷启动样本、1 亿为热物品样本。冷物品定义为过去 7 天曝光数少于 200；热物品定义为过去 7 天点击数大于 3 或订单数大于 0。", styles["BodyCN"]))
offline = [
    [p("方法", styles["TableHeadCN"]), p("All AUC", styles["TableHeadCN"]), p("All GAUC", styles["TableHeadCN"]), p("Cold AUC", styles["TableHeadCN"]), p("Cold GAUC", styles["TableHeadCN"])],
    [p("SPM_SID", styles["TableCN"]), p("0.8663", styles["TableCN"]), p("0.6322", styles["TableCN"]), p("0.8400", styles["TableCN"]), p("0.6203", styles["TableCN"])],
    [p("DAS", styles["TableCN"]), p("0.8679", styles["TableCN"]), p("0.6352", styles["TableCN"]), p("0.8426", styles["TableCN"]), p("0.6237", styles["TableCN"])],
    [p("SaviorRec", styles["TableCN"]), p("0.8687", styles["TableCN"]), p("0.6360", styles["TableCN"]), p("0.8435", styles["TableCN"]), p("0.6249", styles["TableCN"])],
    [p("COINS", styles["TableCN"]), p("0.8725", styles["TableCN"]), p("0.6394", styles["TableCN"]), p("0.8528", styles["TableCN"]), p("0.6301", styles["TableCN"])],
]
story.append(table(offline, [34 * mm, 29 * mm, 29 * mm, 29 * mm, 29 * mm]))
story.append(Spacer(1, 3 * mm))
story.append(p("相较 SaviorRec，COINS 的整体 AUC/GAUC 分别提升 0.38/0.34 个百分点；冷物品 AUC/GAUC 分别提升 0.93/0.52 个百分点。冷域收益大于整体收益，和论文的冷启动目标一致。", styles["BodyCN"]))
ablation = [
    [p("模型", styles["TableHeadCN"]), p("Cold AUC", styles["TableHeadCN"]), p("Cold GAUC", styles["TableHeadCN"]), p("说明", styles["TableHeadCN"])],
    [p("only SID", styles["TableCN"]), p("0.8391", styles["TableCN"]), p("0.6192", styles["TableCN"]), p("仅使用 Semantic ID", styles["TableCN"])],
    [p("IID + SID", styles["TableCN"]), p("0.8401", styles["TableCN"]), p("0.6215", styles["TableCN"]), p("保留随机 ID，并将 SID 作为特征", styles["TableCN"])],
    [p("IID + RQ", styles["TableCN"]), p("0.8479", styles["TableCN"]), p("0.6277", styles["TableCN"]), p("验证共享信息迁移", styles["TableCN"])],
    [p("IID + OPQ", styles["TableCN"]), p("0.8455", styles["TableCN"]), p("0.6257", styles["TableCN"]), p("验证个体差异学习", styles["TableCN"])],
    [p("COINS", styles["TableCN"]), p("0.8528", styles["TableCN"]), p("0.6301", styles["TableCN"]), p("RQ + OPQ + Gate", styles["TableCN"])],
]
story.append(p("消融实验", styles["H2CN"]))
story.append(table(ablation, [29 * mm, 25 * mm, 27 * mm, 69 * mm]))
story.append(p("RQ 单独使用的收益大于 OPQ 单独使用，但两者联合后达到最佳效果，说明共享信息和个体差异具有互补性。", styles["BodyCN"]))
story.append(p("线上 A/B 测试（14 天）", styles["H2CN"]))
online = [
    [p("场景", styles["TableHeadCN"]), p("买家数", styles["TableHeadCN"]), p("订单量", styles["TableHeadCN"])],
    [p("整体", styles["TableCN"]), p("+1.720%", styles["TableCN"]), p("+2.230%", styles["TableCN"])],
    [p("冷启动", styles["TableCN"]), p("+3.512%", styles["TableCN"]), p("+9.639%", styles["TableCN"])],
]
story.append(table(online, [55 * mm, 55 * mm, 55 * mm]))
story.append(PageBreak())

story.append(p("5. 批判性分析", styles["H1CN"]))
story.append(p("优势", styles["H2CN"]))
for text in [
    "把内容和协同信息的不对称性显式建模，而不是简单拼接。",
    "RQ/OPQ 的分工清晰：RQ 负责迁移共性，OPQ 负责保持个性。",
    "实验覆盖大规模工业日志和线上 A/B 测试，商业指标提升增强了方法的应用说服力。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("局限与风险", styles["H2CN"]))
for text in [
    "RQ-OPQ 编码继承自 OneSearch，但本文没有完整披露 codebook 训练、量化误差、更新周期和线上延迟。",
    "对比基线数量较少，尚未充分覆盖近期 Meta-learning、Popularity-aware 和生成式冷启动方法。",
    "OPQ 正样本由 codebook 相似度构造，不一定等价于真实用户偏好相似，可能引入错误正样本。",
    "论文没有展示门控权重在不同物品年龄和流量分层下的分布，也没有报告线上 A/B 的置信区间和 p 值。",
    "冷物品的 7 天曝光统计与测试边界需要进一步核查，以排除潜在时间泄漏。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("证据质量判断", styles["H2CN"]))
story.append(p("离线结果、消融结果和线上订单量提升共同支持 COINS 对冷启动表示有效；但论文更像工业系统报告，缺少跨平台数据、超参数敏感性、门控行为分析和延迟/资源开销。因此可以确认“在该平台和该数据分布上有效”，但尚不能直接推出“在所有冷启动推荐场景中普遍有效”。", styles["BodyCN"]))
story.append(PageBreak())

story.append(p("6. 对课程推荐冷启动的启发", styles["H1CN"]))
story.append(p("COINS 的结构可以自然迁移到课程推荐：把课程内容、知识点和学习路径生成 RQ 共享语义，把教师风格、难度、时长、项目实践和先修要求编码为 OPQ 个体差异，再用门控机制决定当前学生更应该依赖课程行为还是课程内容。", styles["BodyCN"]))
course = [
    [p("COINS 组件", styles["TableHeadCN"]), p("课程推荐对应物", styles["TableHeadCN"]), p("研究问题", styles["TableHeadCN"])],
    [p("RQ", styles["TableCN"]), p("标题、简介、知识点、教材章节、字幕", styles["TableCN"]), p("哪些课程共享知识主题或学习路径？", styles["TableCN"])],
    [p("OPQ", styles["TableCN"]), p("教师、难度、时长、项目、授课风格", styles["TableCN"]), p("同一主题下课程之间有什么差异？", styles["TableCN"])],
    [p("Adaptive Gate", styles["TableCN"]), p("学生画像、查询、学习阶段、课程年龄", styles["TableCN"]), p("新课程应更相信内容还是行为？", styles["TableCN"])],
]
story.append(table(course, [38 * mm, 63 * mm, 64 * mm]))
story.append(p("建议的复现路线", styles["H2CN"]))
for text in [
    "使用预训练语言模型生成课程文本 embedding，并用多层向量量化构造课程 Semantic ID。",
    "保留原始课程 ID embedding，使用门控网络动态融合课程 ID 和语义 ID。",
    "用 KL 迁移损失完成内容-协同信息的方向性迁移，用 InfoNCE 学习课程个体差异。",
    "严格按课程发布时间切分训练和测试，分别报告成熟课程与新课程的 AUC、GAUC、NDCG、Recall 和完成率。",
    "构造 OPQ 正样本时加入知识点、难度和学习路径约束，避免仅按语义相似度造成课程同质化。",
]:
    story.append(bullet(text, styles["BulletCN"]))
story.append(p("最终研究判断", styles["H2CN"]))
story.append(p("COINS 最值得借鉴的不是某个具体量化器，而是“共享信息迁移 + 个体差异保持 + 状态自适应融合”的设计范式。对于课程冷启动，下一步可以进一步加入课程知识图谱、学生能力估计和先修关系，使 Semantic ID 不仅表示课程内容，还能表示课程在学习路径中的位置。", styles["BodyCN"]))
story.append(Spacer(1, 5 * mm))
story.append(p("来源：Qihang Zhao et al., COINS: Semantic Ids Enhanced Cold Item Representation for Click-through Rate Prediction in E-commerce Search, WWW 2026. DOI: 10.1145/3774904.3792902。本文中的数值、公式和实验描述均根据论文正文 Table 1-3、Equation (1)-(9) 整理。", styles["SmallCN"]))


doc = CoinsDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=18 * mm,
    rightMargin=18 * mm,
    topMargin=23 * mm,
    bottomMargin=17 * mm,
    title="COINS 论文精读摘要",
    author="Codex",
    subject="COINS cold-start recommendation paper summary",
)
doc.build(story)
print(f"Generated: {OUT}")
print(f"Bytes: {OUT.stat().st_size}")
