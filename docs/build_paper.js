// Builds the IEEE-style paper (.docx). Tables are generated from results/*.json so numbers cannot drift.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, WidthType,
  BorderStyle, SectionType, ShadingType, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const R = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", f), "utf8"));
const summary = R("summary.json"), sub = R("subgroups.json"), miss = R("missing_modality.json");
const real = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "real", "real_results.json"), "utf8"));
const ds = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "dataset_stats.json"), "utf8"));
const FIG = (f) => fs.readFileSync(path.join(ROOT, "results", "figures", f));

const FONT = "Times New Roman";
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 900, right: 900 } };
const FULL = 12240 - 1800; // 10440 dxa text width
const COLW = (FULL - 360) / 2; // 5040

// ---------- helpers ----------
function runs(text, base = {}) {
  // tiny markup: **bold**, *italic*
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const body = (t) => new Paragraph({ children: runs(t, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 200 }, spacing: { after: 80, line: 228 } });
const h1 = (t) => new Paragraph({ children: [new TextRun({ text: t.toUpperCase(), font: FONT, size: 20, smallCaps: false })],
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true });
const h2 = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, font: FONT, size: 20 })],
  spacing: { before: 100, after: 60 }, keepNext: true });
const cap = (t, wide) => new Paragraph({ children: runs(t, { size: 16 }), alignment: wide ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
  spacing: { before: 40, after: 120 } });
const tcap = (label, t) => new Paragraph({ children: [new TextRun({ text: label, font: FONT, size: 16 }), new TextRun({ text: "\n", break: 1 }), ...runs(t, { size: 16, allCaps: false })],
  alignment: AlignmentType.CENTER, spacing: { before: 100, after: 60 }, keepNext: true });
const ref = (n, t) => new Paragraph({ children: runs(`[${n}]  ${t}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED,
  indent: { left: 360, hanging: 360 }, spacing: { after: 30 } });
const bullet = (t) => new Paragraph({ children: runs(t, { size: 20 }), numbering: { reference: "b", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 228 } });

function img(file, wIn, pxW, pxH) {
  const w = Math.round(wIn * 96), h = Math.round((w * pxH) / pxW);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: FIG(file), transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })] });
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, boldRow, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: none, bottom: last ? thin : none, left: none, right: none },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size: opts.size || 15, bold: head || boldRow }) })],
  });
  const trs = [];
  trs.push(new TableRow({ tableHeader: true, children: header.map((h, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: thin, bottom: thin, left: none, right: none },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(h, { size: opts.size || 15, bold: true }) })] })) }));
  rows.forEach((r, ri) => trs.push(new TableRow({ children: r.cells.map((c, i) => {
    return cell(c, i, false, r.bold, ri === rows.length - 1);
  }) })));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs,
    alignment: AlignmentType.CENTER });
}
const f1 = (t) => `${t[0].toFixed(1)}±${t[1].toFixed(1)}`;
const NAMES = { text: "Text-only", image: "Image-only", late: "Late fusion", concat: "Concat fusion",
  cross: "Cross-modal (ours)", cross_noasp: "  w/o aspect-cond.", cross_nomd: "  w/o modality drop." };

// ---------- tables ----------
const setTable = (split) => table(
  ["Model", "Asp Acc", "Asp P", "Asp R", "Asp F1", "Sen Acc", "Sen P", "Sen R", "Sen F1", "Joint"],
  Object.keys(NAMES).filter((n) => summary[split][n]).map((n) => ({
    bold: n === "cross",
    cells: [NAMES[n], ...["aspect_acc", "aspect_precision", "aspect_recall", "aspect_f1", "sentiment_acc",
      "sentiment_precision", "sentiment_recall", "sentiment_f1", "joint_acc"].map((k) => f1(summary[split][n][k]))],
  })),
  [1900, 950, 950, 950, 950, 950, 950, 950, 950, 940], { size: 15 });

const subTable = () => table(
  ["Model", "Asp: text cue", "Asp: image cue", "Asp: both", "Sen: text cue", "Sen: image cue", "Sen: both", "Conflict"],
  ["text", "image", "late", "concat", "cross"].map((n) => ({ cells: [NAMES[n],
    ...["T", "I", "B"].map((g) => sub[n].aspect[g].toFixed(1)), ...["T", "I", "B"].map((g) => sub[n].sent[g].toFixed(1)),
    sub[n].conflict.toFixed(1)] })),
  [1900, 1150, 1150, 1100, 1150, 1150, 1100, 1000], { size: 15 });

const missTable = () => table(
  ["Model", "Both (A/S)", "Text missing (A/S)", "Image missing (A/S)"],
  ["concat", "cross", "cross_noasp", "cross_nomd"].map((n) => ({ cells: [NAMES[n],
    ...["test_iid", "test_iid_missing_text", "test_iid_missing_image"].map((k) => `${miss[n][k][0].toFixed(1)}/${miss[n][k][1].toFixed(1)}`)] })),
  [1500, 900, 1300, 1300], { size: 14 });

const statTable = () => {
  const r = (k, lab) => ({ cells: [lab, ds[k].n, ds[k].aspect_source.T + "/" + ds[k].aspect_source.I + "/" + ds[k].aspect_source.B,
    ds[k].conflicts, ds[k].avg_words] });
  return table(["Split", "N", "Aspect cue T/I/B", "Conflicts", "Words"],
    [r("train", "Train"), r("val", "Val"), r("test_iid", "Test-IID"), r("test_ood", "Test-OOD")], [1000, 600, 1500, 900, 850], { size: 15 });
};
const datasetsTable = () => table(["Dataset", "Language", "Aspect labels", "Sentiment"], [
  { cells: ["Twitter-15/17 [6]", "English", "Explicit entity target", "neg/neu/pos"] },
  { cells: ["Multi-ZOL [7]", "Chinese", "6 categories (not in-text)", "1–10 score"] },
  { cells: ["MASAD [8]", "English", "57 categories", "neg/pos"] },
  { cells: ["MACSA [10]", "Chinese", "Categories, incl. implicit", "polarity"] },
  { cells: ["ImplicitMABSA-Synth", "English", "8 categories, never named", "neg/neu/pos"] },
], [1700, 850, 1800, 900], { size: 14 });

// ---------- content ----------
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({
    text: "ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis Using Text and Images", font: FONT, size: 44, bold: false })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: "Author Name", font: FONT, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: "[Department, Institution — please edit]", font: FONT, size: 18, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: "author@example.com", font: FONT, size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Project supervisor: Dr. Amit Chauhan, Bennett University", font: FONT, size: 18, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 228 }, children: [
    new TextRun({ text: "Abstract—", font: FONT, size: 18, bold: true, italics: true }),
    ...runs("Aspect-based sentiment analysis (ABSA) usually assumes that the aspect is written in the text. In social-media posts the aspect is often implicit (“Dead by lunch again” is about the battery) and may be clarified only by an attached image. We study **multimodal implicit ABSA**: given a short text and an image, predict the implicit aspect category and its sentiment. No public corpus we could obtain combines implicit aspects, images and sentiment, so we release **ImplicitMABSA-Synth**, a reproducible generated benchmark of 8,000 image–text pairs whose aspects are never named in the text. On it, text-only and image-only models reach aspect macro-F1 of 75.1 and 69.7 and sentiment macro-F1 of 79.7 and 73.4, while every fusion model reaches 100.0 and about 98; the proposed cross-modal attention model is no better than simple concatenation, and all from-scratch models degrade on unseen wording. Because the synthetic cues are complementary by construction, we also evaluate on the real Twitter-2015 and Twitter-2017 corpora (sentiment only, frozen pretrained encoders, target shown or masked). There, text dominates, image-only is close to the majority-class baseline, and fusion brings no reliable gain over text. We discuss these contrasting results and the limits of each benchmark.", { size: 18, italics: false }) ] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [
    new TextRun({ text: "Index Terms—", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "aspect-based sentiment analysis, implicit aspects, multimodal fusion, cross-modal attention, social media", font: FONT, size: 18, italics: true })] }),
];

const secA = [
  h1("I. Introduction"),
  body("Aspect-based sentiment analysis (ABSA) aims to identify the sentiment expressed toward specific aspects of an entity [1], [2]. Most work assumes the aspect term appears in the sentence. In practice, users often express opinions about an aspect without naming it: “The battery lasted only two hours” names the battery, but “Dead by lunch again” does not, and a text-only model must guess the target from context. On social media the situation is harder because posts pair short, informal text with an image. “Not what I expected!” together with a photo of a crushed parcel is negative about *packaging*, yet neither modality alone reveals both the aspect and the polarity."),
  body("This project studies **multimodal implicit ABSA** and addresses the research question: *how effectively can the combination of textual and visual information improve the detection of implicit aspects and their associated sentiment?* We compare text-only, image-only and multimodal models with accuracy, precision, recall and F1."),
  body("**Contributions.** (1) We formalise the task and provide ImplicitMABSA-Synth, a reproducible benchmark whose implicitness is enforced by tests and whose modality informativeness is controlled. (2) We present a controlled comparison of unimodal, late-fusion, concatenation and cross-modal attention models, including analysis by cue location, robustness to a missing modality and to unseen wording. (3) We add a real-data check on Twitter-2015/2017 (sentiment only, target shown or masked) with frozen pretrained encoders. (4) We report negative and limiting findings openly: a more elaborate fusion model does not beat concatenation on the synthetic benchmark, from-scratch encoders do not generalise to unseen phrasings, and on real tweets fusion does not reliably beat text alone. Code, data generator, feature extraction and figures are released."),
  h1("II. Related Work"),
  body("**ABSA.** Classical ABSA benchmarks such as SemEval-2014 Task 4 [1] label aspect terms and categories in review text, and early feature-based mining [3] already recognised that some product features are only implicitly mentioned; rule-based and association-based methods were proposed to recover them [17]. Neural sequence models [12] and Transformers [13], [14] later dominated the field."),
  body("**Multimodal ABSA.** Zhang et al. [4] and Lu et al. [5] collected tweets with images for entity recognition; the resulting Twitter-2015 and Twitter-2017 corpora were extended with sentiment for target-oriented multimodal sentiment classification by Yu and Jiang [6], who adapted BERT with target attention. Xu et al. [7] built Multi-ZOL, Chinese smartphone reviews with images and per-aspect ratings; Zhou et al. [8] released MASAD, a large corpus with 57 aspect categories and binary polarity; Ling et al. [9] proposed vision-language pre-training for the joint task; and Yang et al. [10] introduced MACSA, a Chinese aspect-category dataset with fine-grained aligned image and text annotations that explicitly targets categories not stated in text. Table I summarises these resources. Twitter-15/17 target explicit entity mentions, whereas category-level corpora are hosted on services we could not reach. Multimodal fusion in general is surveyed in [11]."),
  tcap("TABLE I", "Public multimodal ABSA resources and the benchmark used here"),
  datasetsTable(),
  new Paragraph({ spacing: { after: 80 }, children: [] }),
  h1("III. Task and Dataset"),
  h2("A. Task definition"),
  body("Each example is a triple (*t*, *v*, *y*) of a short text *t*, an image *v*, and labels *y* = (*a*, *s*) where *a* ∈ {battery, screen, camera, price, delivery, packaging, sound, performance} is the implicit aspect and *s* ∈ {negative, neutral, positive}. The aspect word never occurs in *t*. Models output both labels."),
  h2("B. Why a generated benchmark"),
  body("No public corpus we could obtain combines implicit aspect categories, images and three-way sentiment. Twitter-15/17 [6] provide images and three-way sentiment but label named-entity targets, not aspect categories; the category corpora [7], [8], [10] are distributed through Google Drive, Baidu or gated hosts that were inaccessible from the build environment. To make the study reproducible and to make the source of information *controllable*, we generate the data programmatically. This is a deliberate trade-off: it gives exact control and ground truth, but the numbers must not be read as real-world accuracy (Sections VII and VIII); Section VII therefore adds a real-data check."),
  h2("C. Generation procedure"),
  body("**Text.** For each aspect and sentiment we wrote eight opinion phrases that avoid all aspect terms (e.g. “Gone from full to almost nothing in an hour” for battery/negative), four aspect-topic phrases without polarity, eight aspect-agnostic sentiment phrases (“Not what I expected!”) and eight uninformative fillers. Random openers/closers, emojis (35% of sentiment-bearing texts, 15% of which are misleading) and typos (4%) are added. A unit test scans every phrase for explicit aspect terms and failed during development until offending phrases (e.g. those containing “fast”) were rewritten."),
  body("**Image.** A procedural renderer draws an aspect icon (battery, phone, camera, price tag, truck, box, speaker, gauge) whose *state* encodes sentiment (empty/full battery, cracked/vivid screen, crushed/neat box, red/green arrow, and so on). Alternatively it draws a reaction face (sentiment only), a grey “ambiguous” icon (aspect only), or clutter (no cue). Position, scale, rotation, background gradients, distractor blobs, noise and blur are randomised; images are 64×64 (Fig. 1)."),
];

const figIcons = [
  img("fig_dataset_icons.png", 6.2, 897, 396),
  cap("Fig. 1. Aspect icons (columns) in negative/neutral/positive states (rows) before the random background, clutter and noise variations used for the dataset.", true),
];

const secB = [
  body("**Information sources.** For every example the aspect is drawn from {text only, image only, both} with probabilities (0.35, 0.30, 0.35) and the sentiment likewise with (0.30, 0.30, 0.40); the corresponding modality receives an informative cue, the other an uninformative one. Hence each example is solvable from at least one modality, but no single modality solves the whole set. When both modalities carry sentiment, in 10% of cases the image contradicts the text and the label follows the text; 2% of sentiment labels are flipped at random."),
  body("**Splits.** 5,600/800/800/800 examples for train/validation/test-IID/test-OOD (Table II). The last two phrases of every list, plus new openers and closers, are used only in test-OOD to measure generalisation to unseen wording."),
  tcap("TABLE II", "Dataset splits. Aspect cue T/I/B: number of examples whose aspect is carried by text only / image only / both."),
  statTable(),
  new Paragraph({ spacing: { after: 80 }, children: [] }),
  h1("IV. Methods"),
  body("All models are trained from scratch; no pretrained weights were available offline. Every model has an aspect head (8 classes) and a sentiment head (3 classes) and is trained with the sum of two cross-entropy losses (label smoothing 0.05)."),
  h2("A. Unimodal encoders"),
  body("**Text:** lower-cased word tokens (emoji kept as tokens) → 96-d embeddings → bidirectional GRU [12] (96 units per direction) → attention pooling (0.24 M parameters with heads). **Image:** four Conv–BatchNorm–ReLU–MaxPool blocks with 16/32/64/64 channels, giving a 4×4 grid of 64-d region features, mean-pooled for the image-only model (0.08 M parameters)."),
  h2("B. Fusion baselines"),
  body("**Late fusion** averages the class probabilities of the separately trained text and image models. **Concat fusion** concatenates the pooled text and image vectors and applies an MLP (0.32 M parameters)."),
  h2("C. Cross-modal attention model (proposed)"),
  body("Word states and image-region features are projected to a shared 96-d space (region positions are learned). Two 4-head attention blocks let words attend to image regions and regions attend to words, each with a residual connection and layer normalisation. The resulting text and image contexts *t* and *v* are combined by a learned gate, *f* = *g*⊙*t* + (1−*g*)⊙*v*, where *g* = σ(*W*[*t*; *v*; *t*⊙*v*; |*t*−*v*|]). The aspect head reads [*f*; *t*; *v*]; the **sentiment head additionally receives the predicted aspect distribution**, so polarity is estimated for the inferred aspect. During training one modality is zeroed for 7.5% of samples each (**modality dropout**). The model has 0.46 M parameters. Two ablations remove aspect-conditioning or modality dropout."),
  h1("V. Experimental Setup"),
  body("Optimiser AdamW (learning rate 2×10⁻³, weight decay 10⁻⁴, one-cycle schedule), batch 64, 15 epochs, ±3 px random translation of images and 10% word dropout; the epoch with the best mean validation macro-F1 is kept. Each configuration is run with three seeds (0–2) and we report mean ± standard deviation. Metrics are accuracy and macro-averaged precision, recall and F1 for each task, plus *joint accuracy* (both labels correct). Runs took about 2.5 min each on two CPU cores."),
];

const wideResults = [
  tcap("TABLE III", "Results on test-IID (top) and test-OOD (bottom), % mean±std over 3 seeds. Asp = implicit aspect, Sen = sentiment; joint = both correct."),
  setTable("test_iid"),
  new Paragraph({ spacing: { after: 60 }, children: [] }),
  setTable("test_ood"),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
  img("fig_main_f1.png", 5.6, 944, 473),
  cap("Fig. 2. Macro-F1 of each model on IID (seen wording) and OOD (unseen wording) test sets; error bars are standard deviations over three seeds.", true),
];

const secC = [
  h1("VI. Results and Discussion"),
  h2("A. Modalities are complementary"),
  body(`On test-IID (Table III) the text-only model reaches ${f1(summary.test_iid.text.aspect_f1)} aspect F1 and ${f1(summary.test_iid.text.sentiment_f1)} sentiment F1, and the image-only model ${f1(summary.test_iid.image.aspect_f1)} and ${f1(summary.test_iid.image.sentiment_f1)}. Every fusion model reaches 100.0 aspect F1 and about 98 sentiment F1, close to the roughly 98% ceiling set by the 2% label noise. The subgroup analysis (Table IV, Fig. 3) shows why: the text model is perfect when the text carries the aspect but at chance (${sub.text.aspect.I.toFixed(1)}% vs 12.5%) when only the image does, and the image model shows the mirror pattern (${sub.image.aspect.T.toFixed(1)}% when only the text does). Fusion recovers both cases. **This complementarity is built into the generator**, so the size of the gain confirms the pipeline works rather than establishing how large it would be on real posts.`),
  body("Fig. 5 shows examples where both unimodal models fail but fusion succeeds; for instance the generic text “Not what I expected!” gives negative polarity while the image supplies the aspect. The command-line demo reproduces this behaviour on new inputs."),
];

const wideSub = [
  tcap("TABLE IV", "IID accuracy (%) by where the cue is available (T = text-only, I = image-only, both) and on text–image sentiment conflicts."),
  subTable(),
  new Paragraph({ spacing: { after: 80 }, children: [] }),
  img("fig_subgroups.png", 5.4, 944, 476),
  cap("Fig. 3. Accuracy by cue location. Each unimodal model fails exactly where its modality carries no cue; fusion models do not.", true),
];

const secD = [
  h2("B. Proposed cross-modal model versus simple fusion"),
  body(`The cross-modal model is **not better** than the simpler baselines. On test-IID it matches concatenation (aspect 100.0 vs 100.0; sentiment ${f1(summary.test_iid.cross.sentiment_f1)} vs ${f1(summary.test_iid.concat.sentiment_f1)}) and late fusion (${f1(summary.test_iid.late.sentiment_f1)}). On test-OOD it is worse (aspect ${f1(summary.test_ood.cross.aspect_f1)} vs ${f1(summary.test_ood.concat.aspect_f1)}; sentiment ${f1(summary.test_ood.cross.sentiment_f1)} vs ${f1(summary.test_ood.concat.sentiment_f1)}), although with three seeds and standard deviations of 2–3 points only the sentiment gap should be treated as a trend. Removing aspect-conditioning or modality dropout changes IID results by at most 0.3 points (Table III), which is within seed noise; modality dropout also did not improve robustness to a missing modality (Table V). We therefore cannot claim a benefit of these components on this benchmark. A plausible reason is that the task is easy enough for a linear combination of pooled features to saturate, leaving no headroom for interaction modelling; real posts with subtle image–text relations may differ.`),
  h2("C. Conflicts and missing modality"),
  body(`When the image contradicts the text, the label follows the text. Text-only and concatenation both reach ${sub.concat.conflict.toFixed(1)}%, the cross-modal model ${sub.cross.conflict.toFixed(1)}%, and late fusion only ${sub.late.conflict.toFixed(1)}% because averaging probabilities lets the image dilute the text; the image-only model is nearly always wrong (${sub.image.conflict.toFixed(1)}%). If one modality is removed at test time by zeroing (Table V), fusion models fall to roughly the level of the surviving unimodal model (about 70–79 F1), i.e. they do not degrade gracefully beyond what the remaining modality can explain.`),
  tcap("TABLE V", "Missing-modality test on IID data: aspect/sentiment macro-F1 (%), mean of three seeds."),
  missTable(),
  new Paragraph({ spacing: { after: 80 }, children: [] }),
  h2("D. Unseen wording"),
  body(`On test-OOD the text-only model collapses (${f1(summary.test_ood.text.aspect_f1)} aspect F1, near the 12.5% chance level; ${f1(summary.test_ood.text.sentiment_f1)} sentiment), because words in the held-out phrases are unknown to an embedding layer trained only on seen phrases, and no pretrained language model was available offline. Image-only performance is unchanged (${f1(summary.test_ood.image.aspect_f1)}), so the visual channel keeps every fusion model well above text-only. Fusion models fall to the level of image-only (about 65–77 F1) rather than exceeding it: once the text is unintelligible they inherit the image’s partial information. Late fusion is marginally best on OOD, since an uninformative text posterior barely disturbs the image posterior. Pretrained text (e.g. BERT [14]) and vision–language encoders are the obvious remedy and are the first item of future work.`),
];

const tableReal = () => {
  const conds = [["2015_explicit", "T15 shown"], ["2015_masked", "T15 masked"], ["2017_explicit", "T17 shown"], ["2017_masked", "T17 masked"]];
  const order = [["majority", "Majority class"], ["text", "Text-only"], ["image", "Image-only"], ["late", "Late fusion"], ["concat", "Concat fusion"], ["cross", "Cross-modal"]];
  const g = (v) => (Array.isArray(v) ? `${v[0].toFixed(1)}±${v[1].toFixed(1)}` : v.toFixed(1));
  const header = ["Model", ...conds.flatMap(([, l]) => [l + " Acc", l + " F1"])];
  const rows = order.map(([k, n]) => ({ cells: [n, ...conds.flatMap(([c]) => [g(real[c][k].acc), g(real[c][k].f1)])] }));
  return table(header, rows, [1500, ...Array(8).fill(1118)], { size: 14 });
};
const num = (v) => (Array.isArray(v) ? v[0] : v).toFixed(1);
const realAcc = (c, k) => num(real[c][k].acc);
const realF1 = (c, k) => num(real[c][k].f1);

const secReal = [
  h1("VII. Real-Data Experiment"),
  h2("A. Data and protocol"),
  body("To test whether the synthetic findings transfer, we also evaluate on the real Twitter-2015 and Twitter-2017 target-oriented multimodal sentiment corpora [4]–[6]. Each instance is a tweet, its image and a target term with a negative, neutral or positive label. We use the standard splits as provided in a public GitHub copy of the data [21] (3,179/1,122/1,037 and 3,562/1,176/1,234 train/dev/test instances; the test class counts match published statistics). These corpora label **named-entity targets** and contain no aspect categories, so **only the sentiment task is evaluated here**; implicit-aspect classification remains evaluated on the synthetic benchmark alone."),
  body("We use two text conditions. In *target shown* the target term replaces the $T$ placeholder, the usual setting. In *target masked* it is replaced by a MASK token, so the model must infer what the sentiment concerns from the remaining context and the image, an implicit-target analogue of our task. Because the corpora are small, the encoders are frozen and pretrained: 300-d word vectors from the spaCy en_core_web_md model [18] feed the same BiGRU with attention, and an ImageNet-pretrained MobileNetV3-Large [19], [20] supplies a 3×3 grid of 960-d region features. The five systems (text, image, late, concat, cross-modal) are the same as before with the aspect head removed. Training uses AdamW (learning rate 10⁻³, weight decay 10⁻²), batch 64, 12 epochs, label smoothing 0.1, model selection on dev macro-F1 and three seeds; a majority-class baseline is included."),
];
const wideReal = [
  tcap("TABLE VI", "Real data: sentiment accuracy and macro-F1 (%) on the Twitter-2015 (T15) and Twitter-2017 (T17) test sets, mean±std over 3 seeds, with the target term shown or masked."),
  tableReal(),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
  img("fig_real.png", 5.6, 1070, 450),
  cap("Fig. 6. Real-data results. Text carries the signal; image-only stays close to the majority baseline and fusion does not reliably beat text.", true),
];
const secReal2 = [
  h2("B. Findings"),
  body(`**Text carries the signal.** Text-only reaches ${realAcc("2015_explicit","text")}% / ${realAcc("2015_masked","text")}% accuracy on Twitter-2015 (majority class: ${realAcc("2015_explicit","majority")}%) and ${realAcc("2017_explicit","text")}% / ${realAcc("2017_masked","text")}% on Twitter-2017 (majority: ${realAcc("2017_explicit","majority")}%), for target shown / masked. **The image contributes little.** Image-only accuracy is ${realAcc("2015_explicit","image")}% on Twitter-2015, below the majority baseline although its macro-F1 (${realF1("2015_explicit","image")}) is above the majority macro-F1 (${realF1("2015_explicit","majority")}) because it predicts minority classes, and ${realAcc("2017_explicit","image")}% on Twitter-2017, barely above the ${realAcc("2017_explicit","majority")}% baseline.`),
  body(`**Fusion does not reliably beat text.** Differences between text-only and the fusion models are at most a few points and mostly inside the seed standard deviations; late fusion is clearly worse on Twitter-2015 with the target shown (macro-F1 ${realF1("2015_explicit","late")} vs ${realF1("2015_explicit","text")}). Concatenation has a slightly higher macro-F1 than text on Twitter-2017 (${realF1("2017_explicit","concat")} vs ${realF1("2017_explicit","text")}; ${realF1("2017_masked","concat")} vs ${realF1("2017_masked","text")}), but with three seeds and standard deviations of 1.6–2.4 points we do not claim a gain. The cross-modal model does not outperform concatenation here either.`),
  body(`**Masking the target does not hurt.** Text-only accuracy is higher with the target masked than shown (${realAcc("2015_masked","text")} vs ${realAcc("2015_explicit","text")}; ${realAcc("2017_masked","text")} vs ${realAcc("2017_explicit","text")}). A plausible but untested explanation is that frozen word vectors of rare entity names add noise, while the sentiment is carried by context words; this suggests the target term itself is of limited help to these simple models.`),
  body("**Contrast with the synthetic benchmark.** On ImplicitMABSA-Synth fusion gains roughly 18–30 F1 points over the unimodal models because we built complementary cues into the data. On real tweets the image rarely adds polarity beyond the text, and a frozen, generic ImageNet encoder is not tuned to sentiment; with small data and three seeds we cannot separate these explanations. The real-data result is the more trustworthy indicator of how hard the problem is, while the synthetic result shows what the models can do when the visual cue is informative."),
  h1("VIII. Limitations and Threats to Validity"),
  bullet("**Synthetic data.** Text comes from about 250 hand-written phrase templates and images from geometric icons. Real implicit aspects are more diverse, and real images are photographs. Absolute scores, especially the 100.0 aspect F1, describe this generator and must not be quoted as real-world performance."),
  bullet("**Complementarity by construction.** The gain of fusion over unimodal models follows from how information sources were sampled, so it demonstrates the mechanism but does not measure real-world benefit."),
  bullet("**Limited statistics.** Three seeds; differences of one to three points among fusion models are not significant."),
  bullet("**Synthetic models trained from scratch.** On the synthetic benchmark all encoders are trained from scratch, which explains the poor OOD text results; on real data the encoders are frozen generic features, so results are not comparable with published systems that fine-tune BERT or vision–language models [6], [9]."),
  bullet("**Real data: sentiment only.** Twitter-15/17 label entities rather than aspect categories, so implicit-aspect classification is not evaluated on real data; the category corpora [7], [8], [10] were unreachable. Real-data results use three seeds and a simple frozen-feature setup."),
  h1("IX. Conclusion and Future Work"),
  body("We formalised multimodal implicit ABSA, released a reproducible benchmark with tested implicitness, and compared text-only, image-only and fusion models. Fusion is decisive when cues are split across modalities (aspect F1 75.1/69.7 → 100.0; sentiment F1 79.7/73.4 → ~98), but a cross-modal attention model with aspect conditioning brought no measurable advantage over concatenation, and all from-scratch models are brittle to unseen wording. On real Twitter-2015/2017 sentiment, text dominates and fusion gives no reliable gain over text with frozen encoders, so the large synthetic gain should not be extrapolated. Future work: (i) evaluate implicit-aspect classification on real category corpora such as MASAD, Multi-ZOL and MACSA; (ii) replace encoders with pretrained BERT and CLIP-style vision–language models; (iii) add hard cases where text and image relate subtly or sarcastically; (iv) use human-annotated implicit-aspect posts."),
  h1("Reproducibility"),
  body("Code, generator (seed 13), tests, trained checkpoint, raw per-run metrics and figures are in the accompanying repository; generate_data.py, run_experiments.py and analyze.py reproduce the synthetic results, and extract_features.py, run_real.py and analyze_real.py reproduce the real-data results from the Twitter data [21]."),
  h1("References"),
  ref(1, "M. Pontiki et al., “SemEval-2014 Task 4: Aspect based sentiment analysis,” in *Proc. SemEval*, 2014, pp. 27–35."),
  ref(2, "B. Liu, *Sentiment Analysis and Opinion Mining*. Morgan & Claypool, 2012."),
  ref(3, "M. Hu and B. Liu, “Mining and summarizing customer reviews,” in *Proc. ACM KDD*, 2004, pp. 168–177."),
  ref(4, "Q. Zhang, J. Fu, X. Liu, and X. Huang, “Adaptive co-attention network for named entity recognition in tweets,” in *Proc. AAAI*, 2018."),
  ref(5, "D. Lu, L. Neves, V. Carvalho, N. Zhang, and H. Ji, “Visual attention model for name tagging in multimodal social media,” in *Proc. ACL*, 2018, pp. 1990–1999."),
  ref(6, "J. Yu and J. Jiang, “Adapting BERT for target-oriented multimodal sentiment classification,” in *Proc. IJCAI*, 2019, pp. 5408–5414."),
  ref(7, "N. Xu, W. Mao, and G. Chen, “Multi-interactive memory network for aspect based multimodal sentiment analysis,” in *Proc. AAAI*, 2019, pp. 371–378."),
  ref(8, "J. Zhou, J. Zhao, J. X. Huang, Q. V. Hu, and L. He, “MASAD: A large-scale dataset for multimodal aspect-based sentiment analysis,” *Neurocomputing*, vol. 455, pp. 47–58, 2021."),
  ref(9, "Y. Ling, J. Yu, and R. Xia, “Vision-language pre-training for multimodal aspect-based sentiment analysis,” in *Proc. ACL*, 2022, pp. 2149–2159."),
  ref(10, "H. Yang, Y. Zhao, J. Liu, Y. Wu, and B. Qin, “MACSA: A multimodal aspect-category sentiment analysis dataset with multimodal fine-grained aligned annotations,” *Multimedia Tools Appl.*, 2024."),
  ref(11, "T. Baltrušaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine learning: A survey and taxonomy,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 2, pp. 423–443, 2019."),
  ref(12, "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in *Proc. EMNLP*, 2014, pp. 1724–1734."),
  ref(13, "A. Vaswani et al., “Attention is all you need,” in *Proc. NeurIPS*, 2017, pp. 5998–6008."),
  ref(14, "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *Proc. NAACL-HLT*, 2019, pp. 4171–4186."),
  ref(15, "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. ICLR*, 2019."),
  ref(16, "L. N. Smith and N. Topin, “Super-convergence: Very fast training of neural networks using large learning rates,” arXiv:1708.07120, 2017."),
  ref(17, "S. Poria, E. Cambria, L.-W. Ku, C. Gui, and A. Gelbukh, “A rule-based approach to aspect extraction from product reviews,” in *Proc. SocialNLP*, 2014, pp. 28–37."),
  ref(18, "M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, “spaCy: Industrial-strength natural language processing in Python,” Zenodo, 2020."),
  ref(19, "A. Howard et al., “Searching for MobileNetV3,” in *Proc. IEEE/CVF ICCV*, 2019, pp. 1314–1324."),
  ref(20, "R. Wightman, “PyTorch Image Models,” GitHub repository, 2019."),
  ref(21, "CopotronicRifat, “TwitterDataMABSA: Twitter2015 and Twitter2017 data for multimodal aspect-based sentiment analysis,” GitHub repository (copy of the data of [6])."),
];

const wideQual = [
  img("fig_examples.png", 5.0, 1014, 679),
  cap("Fig. 5. Test-IID examples where both unimodal models err but the cross-modal model is right (text: “truth / text-only / image-only / ours” as aspect/sentiment).", true),
];

// Fig. 4 (confusion) placed wide with quals
const wideConf = [
  img("fig_confusion.png", 5.4, 1042, 556),
  cap("Fig. 4. Row-normalised confusion matrices of the cross-modal model (seed 0, test-IID).", true),
];

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });
const sec = (children, cols, type) => ({
  properties: { type, page: PAGE, column: cols ? { count: 2, space: 360 } : undefined },
  footers: { default: footer }, children });

const doc = new Document({
  creator: "Author Name", title: "ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [
    sec(titleBlock, false),
    sec(secA, true, SectionType.CONTINUOUS),
    sec(figIcons, false, SectionType.CONTINUOUS),
    sec(secB, true, SectionType.CONTINUOUS),
    sec(wideResults, false, SectionType.CONTINUOUS),
    sec(secC, true, SectionType.CONTINUOUS),
    sec([...wideSub, ...wideConf, ...wideQual], false, SectionType.CONTINUOUS),
    sec(secD, true, SectionType.CONTINUOUS),
    sec(secReal, true, SectionType.CONTINUOUS),
    sec(wideReal, false, SectionType.CONTINUOUS),
    sec(secReal2, true, SectionType.CONTINUOUS),
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(__dirname, "ImplicitMABSA_Paper.docx"), b); console.log("ok"); });
