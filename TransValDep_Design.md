# TransVal-Dep: Thiết kế & Blueprint Code

**Một benchmark đa-corpus, kiểm soát confound, calibration-first cho phát hiện trầm cảm từ phỏng vấn lâm sàng**

> Tài liệu này là bản thiết kế kỹ thuật đầy đủ (design doc + code blueprint) để hiện thực hoá hướng nghiên cứu mới phát triển từ bài *"A Leakage-Free Reassessment of Multimodal Depression Detection"*. Mục tiêu: viết được code ngay. Mọi module đều có đặc tả input/output, shape tensor, và ý tưởng code PyTorch cụ thể.

---

## 0. Tóm tắt một trang

| Mục | Nội dung |
|---|---|
| **Tên dự án** | TransVal-Dep (Transferable-Validity Depression detection) |
| **Câu hỏi NC** | Dấu hiệu trầm cảm nào *thực sự là của trầm cảm* (không phải distress chung) và *thực sự chuyển giao được* giữa corpus/ngôn ngữ? |
| **Đóng góp chính** | (1) Benchmark transferable-validity; (2) Đánh giá kiểm soát confound; (3) Học biểu diễn bất biến (adversarial + Group-DRO); (4) Calibration-first + selective prediction; (5) Thống kê đủ power + TOST. |
| **Bỏ khỏi bài cũ** | CUAF context-adaptive fusion, BCM "Bioenergy", reliability=1−entropy. |
| **Giữ & nâng cấp** | Leakage-free participant-level CV, calibration, phát hiện acoustic-dominance (thành giả thuyết). |
| **Dữ liệu** | DAIC-WOZ / E-DAIC (EN) + EATD (ZH) + Androids (IT); tùy chọn CMDC/MODMA. |
| **Metric** | AUC, confound-conditioned AUC, specificity-gap, ECE, Brier, risk–coverage AUC, TOST equivalence. |
| **Venue** | IEEE TAC / Interspeech / EMNLP (resources & evaluation) / Pattern Recognition. |

---

## 1. Năm tính mới (đặc tả để hiện thực hoá)

### NV1 — Transferable-Validity Benchmark
Giao thức đánh giá chuẩn hoá: đa corpus, **participant-level leakage-free**, calibration trên train-fold, confound-stratification. Đóng gói thành "evaluation card" + code release.
- *Hiện thực:* lớp `LeakageFreeSplitter` + `EvaluationCard` (mục 6, 9).
- *Xoá hạn chế:* dữ liệu nhỏ (gộp corpus), external validity, không artifact.

### NV2 — Đánh giá kiểm soát confound (đóng góp đinh)
Partial-out giới tính, tuổi, độ dài phỏng vấn, comorbidity (PTSD/lo âu nếu có). Báo cáo **confound-conditioned AUC** và **specificity-gap** = AUC(depression) − AUC(distress-proxy).
- *Hiện thực:* `ConfoundEvaluator` (mục 8.3), `partial_auc_by_group`, residualization của logit theo confound.
- *Xoá hạn chế:* lỗ hổng validity ([13]), gender bias (Bailey & Plumbley).

### NV3 — Học biểu diễn bất biến / khử bias (thay CUAF/BCM)
Shared depression head + **adversary corpus/gender** qua Gradient Reversal (DANN) + **Group-DRO** (tối ưu nhóm khó nhất). Tuỳ chọn **IRM penalty**.
- *Hiện thực:* `GradientReversal`, `DomainAdversary`, `GroupDROLoss`, `irm_penalty` (mục 5.4–5.6).
- *Xoá hạn chế:* kiến trúc chết + reliability yếu.

### NV4 — Calibration-first + Selective Prediction
Isotonic/Platt trên train-fold; selective head với ngưỡng abstain. Báo cáo risk–coverage, ECE, Brier, decision-curve.
- *Hiện thực:* `ProbabilityCalibrator`, `SelectivePredictor`, `risk_coverage_curve` (mục 5.7, 8.4).
- *Xoá hạn chế:* bẫy F1-một-ngưỡng trên N nhỏ.

### NV5 — Thống kê đủ power + pre-registration
Multi-seed ≥5; participant-level bootstrap CI; **TOST equivalence** trong biên ε; giả thuyết pre-registered.
- *Hiện thực:* `bootstrap_ci`, `tost_equivalence`, `aggregate_seeds` (mục 8.5).
- *Xoá hạn chế:* "n=5 fold không kết luận được".

### (Tái sử dụng) Question-wise segmentation → **validity probe**
Không điều khiển fusion nữa; dùng để đo *câu hỏi nào* mang tín hiệu chuyển giao được & bền confound.

---

## 2. Cấu trúc repository

```
transval-dep/
├── configs/
│   ├── default.yaml            # siêu tham số gốc
│   ├── corpora.yaml            # đường dẫn + schema từng corpus
│   └── experiments/            # 1 file/thí nghiệm (ablation)
├── data/
│   ├── raw/                    # corpus gốc (read-only)
│   ├── interim/                # segment + feature cache (.pt / .npy)
│   └── manifests/              # CSV: participant_id, label, gender, age, corpus, split...
├── src/
│   ├── data/
│   │   ├── segmentation.py     # cắt theo prompt, lọc <3s / >60s
│   │   ├── features.py         # wav2vec2 / HuBERT, BERT, COVAREP, CLNF
│   │   ├── confounds.py        # trích & chuẩn hoá metadata confound
│   │   ├── dataset.py          # SegmentDataset, collate, bag aggregation
│   │   └── splitter.py         # LeakageFreeSplitter (participant-level)
│   ├── models/
│   │   ├── encoders.py         # ModalityEncoder, AttentionPool
│   │   ├── grl.py              # GradientReversal
│   │   ├── adversary.py        # DomainAdversary (corpus/gender)
│   │   ├── heads.py            # DepressionHead, SelectiveHead
│   │   └── transval_net.py     # kiến trúc tổng hợp
│   ├── losses/
│   │   ├── group_dro.py        # GroupDROLoss
│   │   ├── irm.py              # irm_penalty
│   │   └── consistency.py      # (tuỳ chọn) intermodal KL
│   ├── calibration/
│   │   ├── calibrators.py      # Platt, Isotonic
│   │   └── selective.py        # SelectivePredictor, risk-coverage
│   ├── train/
│   │   ├── trainer.py          # vòng huấn luyện + adversarial schedule
│   │   └── cv_runner.py        # 5-fold × multi-seed orchestrator
│   ├── eval/
│   │   ├── metrics.py          # auc, ece, brier, f1, partial_auc
│   │   ├── confound_eval.py    # ConfoundEvaluator, specificity-gap
│   │   ├── stats.py            # bootstrap_ci, tost_equivalence
│   │   └── probe.py            # question-type validity probe
│   └── utils/                  # seed, logging, registry
├── scripts/
│   ├── 00_build_manifests.py
│   ├── 01_extract_features.py
│   ├── 02_run_cv.py
│   └── 03_make_report.py
└── tests/                      # unit test cho splitter, GRL, calibrator
```

---

## 3. Xử lý dữ liệu (chi tiết)

### 3.1 Lược đồ corpus thống nhất (`corpora.yaml`)
Mọi corpus được map về một schema chung để code không phụ thuộc corpus:

```yaml
daic_woz:
  language: en
  audio_dir: data/raw/daic/{pid}/{pid}_AUDIO.wav
  transcript: data/raw/daic/{pid}/{pid}_TRANSCRIPT.csv
  prompts_speaker: "Ellie"          # tên người phỏng vấn để cắt segment
  label_csv: data/raw/daic/labels.csv  # cột: Participant_ID, PHQ8_Binary, PHQ8_Score, Gender
  acoustic: data/raw/daic/{pid}/{pid}_COVAREP.csv
  visual:   data/raw/daic/{pid}/{pid}_CLNF_AUs.txt
eatd:
  language: zh
  audio_dir: data/raw/eatd/{pid}/*.wav
  label_csv: data/raw/eatd/labels.csv
  prompts_speaker: null              # EATD không có cấu trúc Ellie → segment theo VAD
androids:
  language: it
  audio_dir: data/raw/androids/{pid}/interview.wav
  label_csv: data/raw/androids/labels.csv
  prompts_speaker: null
```

### 3.2 Manifest cấp participant (`data/manifests/all.csv`)
Một dòng / participant. **Đây là đơn vị split** (chống leakage).

| cột | kiểu | ý nghĩa |
|---|---|---|
| `participant_id` | str | duy nhất toàn cục: `daic_303`, `eatd_12` |
| `corpus` | str | daic / eatd / androids |
| `language` | str | en / zh / it |
| `label` | int | 0/1 (ngưỡng PHQ-8 ≥ 10 hoặc tương đương của corpus) |
| `severity` | float | PHQ-8/SDS score nếu có (cho phân tích phụ) |
| `gender` | int | 0/1 (confound) |
| `age` | float | nếu có (confound) |
| `interview_len_s` | float | confound (độ dài) |
| `comorbidity_ptsd` | int | nếu có (E-DAIC) |
| `n_segments` | int | điền sau khi segment |

> **Quy tắc nhãn nghiêm ngặt:** nhãn là *cấp participant*. Segment **chỉ là instance**, không gán nhãn cứng — xem MIL/bag ở 3.5.

### 3.3 Segmentation (`segmentation.py`)
- **Có prompt (DAIC):** cắt mỗi đoạn trả lời giữa hai prompt liên tiếp của Ellie; gắn `question_type` từ prompt trước đó qua một bảng ánh xạ `prompt2qtype.yaml` (greeting/background/family/mood/sleep/work/stress/therapy/other). **Bảng ánh xạ phải công khai và versioned** (đây là điểm bài cũ thiếu).
- **Không prompt (EATD/Androids):** dùng VAD (silero-vad) cắt theo turn; `question_type = "unknown"`.
- Lọc: bỏ segment < 3s, cắt cap 60s.

**I/O:**
```
input : audio_path (wav 16kHz), transcript_df, prompt_map
output: List[Segment] với Segment = {
          seg_id, participant_id, start_s, end_s,
          question_type, audio_slice_path | (offset,dur)
        }
```

### 3.4 Trích đặc trưng (`features.py`)
Cache ra `.pt` để không tính lại. **Mỗi modality là một nhánh độc lập** (acoustic là nhánh mạnh nhất theo phát hiện cũ → là baseline bắt buộc).

| Modality | Backbone | Output / segment |
|---|---|---|
| `audio` | wav2vec2 / HuBERT (frozen, đa ngôn ngữ: dùng `wav2vec2-xlsr` cho cross-lingual) | `(T_a, 1024)` chuỗi frame |
| `acoustic` | COVAREP+FORMANT chuẩn hoá z-score | `(T_c, 79)` |
| `text` | BERT/RoBERTa multilingual (frozen) `[CLS]` | `(768,)` |
| `visual` | CLNF AUs+gaze+pose | `(T_v, 50)` (chỉ DAIC; corpus khác → mask) |

> **Cross-lingual lưu ý:** dùng backbone đa ngôn ngữ (XLS-R cho audio, XLM-R cho text). Modality thiếu (vd visual ở EATD) → **modality-mask** trong collate, không impute bừa.

**I/O của một segment sau feature:**
```python
seg_feat = {
  "audio":    Tensor(T_a, 1024),   # hoặc None → mask
  "acoustic": Tensor(T_c, 79),
  "text":     Tensor(768),
  "visual":   Tensor(T_v, 50) | None,
  "qtype":    int,                 # id question_type
  "meta": {"participant_id","corpus","gender","language"}
}
```

### 3.5 Bag (MIL) cấp participant — sửa label-noise của bài cũ
Thay vì gán nhãn session cho từng segment rồi train (gây nhiễu), ta dùng **Multiple-Instance Learning**: participant = bag, segment = instance, nhãn ở **bag**.

```
Bag = {
  segments: List[seg_feat],      # số lượng thay đổi
  label: int,                    # bag-level
  group_id: int                  # (corpus,gender) → cho Group-DRO
}
```
Aggregation từ segment → participant dùng **attention pooling học được** (mục 5.3), KHÔNG dùng mean cứng.

### 3.6 Collate (`dataset.py`)
Batch theo **bag** (participant) để loss bag-level đúng. Pad chuỗi frame trong từng modality, tạo `attention_mask`, và `modality_mask` (1 nếu modality tồn tại).

```python
def collate_bags(bags):
    # gom segment của nhiều bag, ghi lại chỉ số bag để pool lại
    return {
      "audio":        pad_sequence(...),      # (N_seg, T_max, 1024)
      "audio_mask":   BoolTensor(N_seg,T_max),
      "acoustic":     ...,
      "text":         Tensor(N_seg, 768),
      "visual":       ...,
      "modality_mask":Tensor(N_seg, 4),       # tồn tại modality nào
      "seg2bag":      LongTensor(N_seg),       # mỗi segment thuộc bag nào
      "bag_labels":   LongTensor(B),
      "group_ids":    LongTensor(B),
      "corpus_ids":   LongTensor(N_seg),
      "gender_ids":   LongTensor(N_seg),
      "qtypes":       LongTensor(N_seg),
    }
```

---

## 4. Kiến trúc tổng thể

```
                 ┌─────────── per-segment ───────────┐
 audio (T,1024)─▶│ AudioEncoder  → e_a (d)            │
 acoustic(T,79)─▶│ AcousticEncoder→ e_c (d)           │
 text (768)─────▶│ TextEncoder   → e_t (d)            │
 visual(T,50)──▶ │ VisualEncoder → e_v (d) [maskable] │
                 └──────────────┬────────────────────┘
                                │ concat + modality-mask
                          z_seg = FuseMLP([e_a,e_c,e_t,e_v]) (d)
                                │
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
   DepHead(seg)          AttentionPool(seg→bag)        GRL → DomainAdversary
   logit_seg                 z_bag (d)                 (corpus_id, gender_id)
                                │
                          DepHead(bag) → logit_bag ──▶ calibrate ──▶ p_bag
                                │
                          SelectiveHead → g_bag (abstain gate)
```

Hai cấp dự đoán: **segment** (phụ trợ, cho probe & regularize) và **bag** (chính). Adversary gắn ở `z_seg` để khử thông tin corpus/gender khỏi biểu diễn.

---

## 5. Đặc tả & blueprint từng module

### 5.1 AttentionPool (segment-frame → vector)
```python
class AttentionPool(nn.Module):
    """Gộp chuỗi frame (T,d_in) → vector (d_out) bằng attention có mask."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.attn = nn.Linear(d_out, 1)
    def forward(self, x, mask):           # x:(N,T,d_in) mask:(N,T) bool
        h = torch.tanh(self.proj(x))      # (N,T,d_out)
        a = self.attn(h).squeeze(-1)      # (N,T)
        a = a.masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (N,T,1)
        return (w * h).sum(1)             # (N,d_out)
```

### 5.2 ModalityEncoder (mỗi modality một bản)
```python
class ModalityEncoder(nn.Module):
    """
    audio/acoustic/visual: BiLSTM + AttentionPool (chuỗi → d)
    text: MLP ([CLS] → d)
    input : (N,T,F) + mask  | hoặc (N,F) cho text
    output: e (N, d)
    """
    def __init__(self, in_dim, d, seq=True):
        super().__init__()
        self.seq = seq
        if seq:
            self.rnn  = nn.LSTM(in_dim, d//2, batch_first=True, bidirectional=True)
            self.pool = AttentionPool(d, d)
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim,d), nn.GELU(), nn.LayerNorm(d))
    def forward(self, x, mask=None):
        if self.seq:
            h,_ = self.rnn(x)             # (N,T,d)
            return self.pool(h, mask)     # (N,d)
        return self.mlp(x)               # (N,d)
```

### 5.3 Fusion + Bag pooling
- **FuseMLP:** nối 4 embedding (đã nhân `modality_mask` để zero modality thiếu) → MLP → `z_seg (N_seg,d)`. *Lưu ý:* đây là fusion **tĩnh, đơn giản** — KHÔNG context-adaptive (đã chứng minh vô ích). Mọi novelty nằm ở invariance + calibration, không ở fusion.
- **Bag AttentionPool:** gộp các `z_seg` cùng `seg2bag` → `z_bag (B,d)` (attention học được, thay mean cứng của bài cũ).

```python
def pool_segments_to_bags(z_seg, seg2bag, B, attn):
    # attn: Linear(d,1). Pool có mask theo bag.
    scores = attn(z_seg).squeeze(-1)            # (N_seg,)
    z_bag = z_seg.new_zeros(B, z_seg.size(1))
    for b in range(B):                          # vectorise bằng scatter_softmax khi tối ưu
        idx = (seg2bag == b)
        w = torch.softmax(scores[idx], 0).unsqueeze(-1)
        z_bag[b] = (w * z_seg[idx]).sum(0)
    return z_bag
```
> Khi tối ưu tốc độ: thay vòng for bằng `torch_scatter.scatter_softmax`.

### 5.4 GradientReversal (lõi của NV3)
```python
class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None          # đảo dấu gradient

class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0): super().__init__(); self.lambd=lambd
    def forward(self, x): return _GradRev.apply(x, self.lambd)
```
`lambd` tăng dần theo lịch huấn luyện: `λ = 2/(1+exp(-γ·p)) − 1`, `p` ∈ [0,1] tiến trình.

### 5.5 DomainAdversary (corpus & gender)
```python
class DomainAdversary(nn.Module):
    """Dự đoán corpus_id và gender_id từ z_seg SAU GRL → ép z_seg quên chúng."""
    def __init__(self, d, n_corpus, n_gender=2, lambd=1.0):
        super().__init__()
        self.grl = GradientReversal(lambd)
        self.corpus_clf = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,n_corpus))
        self.gender_clf = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,n_gender))
    def forward(self, z):
        z = self.grl(z)
        return self.corpus_clf(z), self.gender_clf(z)   # (N,n_corpus),(N,n_gender)
```
Loss adversary = CE(corpus) + CE(gender). Vì GRL đảo dấu, tối thiểu hoá loss này khiến encoder *xoá* thông tin corpus/gender.

### 5.6 Group-DRO + IRM (NV3, tuỳ chọn)
```python
class GroupDROLoss(nn.Module):
    """Tối ưu nhóm có loss cao nhất (vd nam-trầm-cảm). group_ids∈[0,G)."""
    def __init__(self, n_groups, step=0.01):
        super().__init__(); self.q = torch.ones(n_groups)/n_groups; self.step=step
    def forward(self, per_sample_loss, group_ids):
        G = self.q.numel()
        g_loss = torch.stack([per_sample_loss[group_ids==g].mean()
                              if (group_ids==g).any() else per_sample_loss.new_zeros(())
                              for g in range(G)])
        self.q = (self.q.to(g_loss) * torch.exp(self.step*g_loss.detach()))
        self.q = self.q/self.q.sum()
        return (self.q * g_loss).sum()

def irm_penalty(logits, y, dummy_w):           # dummy_w = torch.tensor(1., requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits*dummy_w, y.float())
    g = torch.autograd.grad(loss, dummy_w, create_graph=True)[0]
    return g.pow(2).sum()                       # cộng vào tổng loss với hệ số nhỏ
```

### 5.7 Heads: Depression + Selective + Calibrator
```python
class DepressionHead(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,1)
    def forward(self,z): return self.fc(z).squeeze(-1)        # logit (B,)

class SelectiveHead(nn.Module):
    """g∈[0,1]: 1=dự đoán, 0=abstain. Huấn luyện kèm phạt coverage."""
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,1)
    def forward(self,z): return torch.sigmoid(self.fc(z)).squeeze(-1)
```
Calibration **không** học trong mạng — fit *sau* khi train, **chỉ trên train-fold** (mục 7):
```python
class ProbabilityCalibrator:
    def __init__(self, method="isotonic"): self.method=method; self.model=None
    def fit(self, logits, y):
        p = 1/(1+np.exp(-logits))
        if self.method=="platt":
            self.model = LogisticRegression().fit(logits.reshape(-1,1), y)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self
    def transform(self, logits):
        p = 1/(1+np.exp(-logits))
        return (self.model.predict_proba(logits.reshape(-1,1))[:,1]
                if self.method=="platt" else self.model.predict(p))
```

### 5.8 TransValNet (lắp ráp)
```python
class TransValNet(nn.Module):
    def __init__(self, d=128, n_corpus=3, use_adv=True, use_visual=True):
        super().__init__()
        self.enc = nn.ModuleDict({
          "audio":    ModalityEncoder(1024,d,seq=True),
          "acoustic": ModalityEncoder(79,  d,seq=True),
          "text":     ModalityEncoder(768, d,seq=False),
          "visual":   ModalityEncoder(50,  d,seq=True),
        })
        self.fuse   = nn.Sequential(nn.Linear(4*d,d), nn.GELU(), nn.LayerNorm(d))
        self.bag_attn = nn.Linear(d,1)
        self.dep_seg = DepressionHead(d)
        self.dep_bag = DepressionHead(d)
        self.selective = SelectiveHead(d)
        self.adv = DomainAdversary(d, n_corpus) if use_adv else None

    def forward(self, batch):
        e = []
        for m in ["audio","acoustic","text","visual"]:
            x = batch[m]
            if m=="text": emb = self.enc[m](x)
            else:         emb = self.enc[m](x, batch[m+"_mask"])
            e.append(emb)
        E = torch.stack(e,1) * batch["modality_mask"].unsqueeze(-1)  # zero modality thiếu
        z_seg = self.fuse(E.flatten(1))                              # (N_seg,d)
        z_bag = pool_segments_to_bags(z_seg, batch["seg2bag"],
                                      batch["bag_labels"].size(0), self.bag_attn)
        out = {
          "logit_bag": self.dep_bag(z_bag),
          "logit_seg": self.dep_seg(z_seg),
          "gate_bag":  self.selective(z_bag),
          "z_seg": z_seg, "z_bag": z_bag,
        }
        if self.adv is not None:
            out["corpus_logit"], out["gender_logit"] = self.adv(z_seg)
        return out
```

---

## 6. Leakage-free splitter (NV1)

```python
class LeakageFreeSplitter:
    """
    5-fold participant-level, stratified theo (label, corpus).
    Đảm bảo: không participant nào ở >1 fold; calibrator/threshold
    chỉ fit trên train-fold; mỗi participant được predict đúng 1 lần.
    """
    def __init__(self, manifest_df, n_folds=5, seed=0):
        self.df=manifest_df; self.k=n_folds; self.seed=seed
    def folds(self):
        # khoá stratify = label*10 + corpus_id  (giữ tỉ lệ dương & corpus mỗi fold)
        skf = StratifiedKFold(self.k, shuffle=True, random_state=self.seed)
        key = self.df["label"]*10 + self.df["corpus_id"]
        for tr, te in skf.split(self.df, key):
            yield self.df.iloc[tr]["participant_id"].tolist(), \
                  self.df.iloc[te]["participant_id"].tolist()
```
> **Hai chế độ đánh giá** trong cùng splitter: (a) *pooled CV* (gộp corpus) cho power; (b) *leave-one-corpus-out* (LOCO) cho transfer thực sự — train trên 2 corpus, test corpus thứ 3. LOCO là phép thử transfer mạnh nhất.

---

## 7. Quy trình Training (`trainer.py`)

### 7.1 Tổng loss
```
L = L_dep_bag                         # BCE bag-level (chính)
  + α · L_dep_seg                     # BCE segment phụ trợ (regularize)
  + β · L_adv (corpus+gender, qua GRL)  # NV3 bất biến
  + γ · L_irm                         # NV3 tuỳ chọn
  + δ · L_consistency                 # tuỳ chọn (KL intermodal) — mặc định 0
```
- `L_dep_bag` dùng **Group-DRO** thay vì BCE thường khi `use_group_dro=True`.
- Class imbalance: `pos_weight` trong BCE **hoặc** Group-DRO (không chồng cả hai để tránh méo probability trước calibration).

### 7.2 Lịch huấn luyện
```python
for epoch in range(E):
    p = epoch/E
    lambd = 2/(1+math.exp(-10*p)) - 1            # warmup adversary
    model.adv.grl.lambd = lambd
    for batch in train_loader:
        out = model(batch)
        per_sample = F.binary_cross_entropy_with_logits(
            out["logit_bag"], batch["bag_labels"].float(), reduction="none")
        L_dep = (group_dro(per_sample, batch["group_ids"])
                 if cfg.use_group_dro else per_sample.mean())
        L_seg = F.binary_cross_entropy_with_logits(
            out["logit_seg"], gather_seg_labels(batch), reduction="mean")  # nhãn bag broadcast cho seg CHỈ ở loss phụ
        L_adv = (F.cross_entropy(out["corpus_logit"], batch["corpus_ids"])
               + F.cross_entropy(out["gender_logit"], batch["gender_ids"]))
        L = L_dep + cfg.alpha*L_seg + cfg.beta*L_adv
        if cfg.use_irm:
            L = L + cfg.gamma*irm_penalty(out["logit_bag"], batch["bag_labels"], dummy_w)
        opt.zero_grad(); L.backward(); clip_grad_norm_(model.parameters(),1.0); opt.step()
    # early stopping theo AUC trên một held-out *trong train-fold* (inner split)
```
> **Backbone đông cứng** (wav2vec2/BERT) → chỉ train encoder nhẹ + heads + adversary.

### 7.3 Calibration sau train (chỉ train-fold)
Sau khi train xong mỗi fold: lấy logit của **inner-validation thuộc train-fold**, fit `ProbabilityCalibrator` + chọn ngưỡng (mặc định 0.5 sau isotonic; hoặc Youden trên inner-val). **Tuyệt đối không** chạm test-fold.

### 7.4 Multi-seed (NV5)
Mỗi fold chạy `seeds = [0,1,2,3,4]`. Lưu mọi prediction. CI cuối cùng phản ánh **cả** fold-variance lẫn seed-variance (sửa lỗi bài cũ).

---

## 8. Quy trình Testing & Đánh giá

### 8.1 Suy luận trên test-fold
Với mỗi participant test: forward → `logit_bag` → calibrator(train-fold) → `p_bag` → ngưỡng(train-fold) → nhãn. Mỗi participant predict đúng 1 lần.

### 8.2 Metric cơ bản (`metrics.py`)
- AUC-ROC (threshold-independent — **primary**), F1/Precision/Recall (ở ngưỡng đã transfer), **Specificity & Sensitivity** (bài cũ hứa nhưng thiếu — phải có).
- **ECE** (calibration error), **Brier score**.

### 8.3 ConfoundEvaluator (NV2 — đóng góp đinh)
```python
class ConfoundEvaluator:
    """
    1) AUC theo từng nhóm confound (gender/corpus/độ dài-binned).
    2) Residualized AUC: hồi quy logit theo confound, đánh giá phần dư.
    3) specificity_gap = AUC(depression) − AUC(distress_proxy)
       distress_proxy: nhãn 'có distress' (PTSD/anxiety/score cao nhưng dưới ngưỡng dep)
    """
    def partial_auc_by_group(self, y, p, group):
        return {g: roc_auc_score(y[group==g], p[group==g]) for g in np.unique(group)}
    def residualized_auc(self, y, logit, C):    # C: ma trận confound (gender,age,len...)
        r = logit - LinearRegression().fit(C, logit).predict(C)
        return roc_auc_score(y, r)
    def specificity_gap(self, y_dep, y_distress, p):
        return roc_auc_score(y_dep, p) - roc_auc_score(y_distress, p)
```
**Câu chuyện:** nếu AUC sụt mạnh sau residualization, hoặc specificity_gap ≈ 0 → mô hình bắt distress/confound chứ không phải trầm cảm. Nếu invariance (NV3) giữ được AUC sau residualization → bằng chứng học được tín hiệu *thật*.

### 8.4 Selective prediction (NV4)
```python
def risk_coverage_curve(y, p, gate):
    order = np.argsort(-gate)                  # ưu tiên giữ ca tự tin
    cov, risk = [], []
    for k in range(1, len(order)+1):
        idx = order[:k]
        yhat = (p[idx] >= 0.5).astype(int)
        cov.append(k/len(order))
        risk.append(1 - f1_score(y[idx], yhat))   # risk = 1−F1 (hoặc error)
    return np.array(cov), np.array(risk)         # báo cáo AURC = area dưới đường
```
Báo cáo **AURC** và F1 tại coverage 70%/50% (triage thực tế). Decision-curve analysis cho ngưỡng lâm sàng.

### 8.5 Thống kê (NV5)
```python
def bootstrap_ci(y, p, metric=roc_auc_score, n=2000, level=0.95):
    rng=np.random.default_rng(0); vals=[]
    idx0=np.arange(len(y))
    for _ in range(n):
        s=rng.choice(idx0, len(idx0), replace=True)   # bootstrap CẤP PARTICIPANT
        try: vals.append(metric(y[s], p[s]))
        except ValueError: pass
    lo,hi=np.percentile(vals,[(1-level)/2*100,(1+level)/2*100])
    return float(np.mean(vals)), (float(lo),float(hi))

def tost_equivalence(diffs, eps):
    """diffs: chênh lệch metric A−B qua các fold×seed. H0: |Δ|≥eps. Reject→tương đương."""
    m,sd,n = diffs.mean(), diffs.std(ddof=1), len(diffs)
    se = sd/np.sqrt(n)
    t_low  = (m - (-eps))/se;  p_low  = 1 - t.cdf(t_low, n-1)
    t_high = (eps - m)/se;     p_high = 1 - t.cdf(t_high, n-1)
    return max(p_low, p_high)   # < 0.05 ⇒ kết luận TƯƠNG ĐƯƠNG (không chỉ "không khác")
```
> Đây là khác biệt sống còn với bài cũ: thay vì "p>0.05 nên không kết luận", ta **chứng minh tương đương trong biên ε** (ε pre-registered, vd ε_AUC=0.03).

### 8.6 Question-type validity probe (`probe.py`)
Dùng `logit_seg` nhóm theo `question_type` × corpus: đo AUC từng loại câu hỏi, kiểm tra loại nào **chuyển giao** (AUC ổn định giữa corpus) và **bền confound** (AUC giữ sau residualization). Kết quả: bảng question_type × {AUC_in, AUC_transfer, AUC_residualized}. Đây là phần "hồi sinh" segmentation theo hướng có giá trị.

---

## 9. Ma trận thí nghiệm (ablation)

| Exp | Mô tả | Kiểm chứng |
|---|---|---|
| E0 | **Acoustic-only** + calibration (baseline bắt buộc) | tham chiếu — mọi thứ phải vượt mốc này |
| E1 | Full fusion, **không** adversary, không Group-DRO | baseline multimodal |
| E2 | E1 + **corpus-adversary** (NV3) | transfer cải thiện? |
| E3 | E2 + **gender-adversary** | confound gender giảm? |
| E4 | E3 + **Group-DRO** | nhóm khó cải thiện? |
| E5 | E4 + **IRM** | bất biến mạnh hơn? |
| E6 | Best + **selective prediction** | AURC, F1@coverage |
| LOCO | Best dưới leave-one-corpus-out | transfer thật |
| PROBE | question-type validity | câu hỏi nào mang tín hiệu thật |

Mỗi exp: 5 fold × 5 seed; báo cáo mean + bootstrap CI; mọi so sánh kèm **TOST**. Mọi exp đều report confound-conditioned AUC + specificity-gap.

---

## 10. Config mẫu (`configs/default.yaml`)

```yaml
model:  {d: 128, use_adv: true, use_group_dro: true, use_irm: false, use_visual: true}
loss:   {alpha: 0.3, beta: 0.5, gamma: 0.1, delta: 0.0, pos_weight: 2.0}
train:  {epochs: 40, lr: 1.0e-3, batch_bags: 8, patience: 10, grad_clip: 1.0}
cv:     {n_folds: 5, seeds: [0,1,2,3,4], mode: pooled}   # pooled | loco
calib:  {method: isotonic, threshold: youden_inner}
stats:  {eps_auc: 0.03, eps_f1: 0.05, bootstrap_n: 2000}
backbones: {audio: facebook/wav2vec2-xls-r-300m, text: xlm-roberta-base}
```

---

## 11. Pre-registration (viết TRƯỚC khi chạy)

1. **H1 (transfer):** corpus-adversary cải thiện LOCO-AUC so với E1, Δ>ε.
2. **H2 (validity):** specificity-gap của model bất biến > 0 và > model baseline.
3. **H3 (fairness):** gender-adversary thu hẹp |AUC_nam − AUC_nữ|.
4. **H_equiv:** nếu fusion không vượt acoustic-only (E0), kết luận **tương đương** qua TOST (ε_AUC=0.03) — biến negative thành phát biểu thống kê hợp lệ.
5. Khoá ε, số seed, số fold, metric chính **trước** khi xem test.

---

## 12. Checklist tái lập (artifact — bài cũ thiếu)

- [ ] Release code + `prompt2qtype.yaml` versioned.
- [ ] Lưu danh sách participant từng fold/seed (`splits/`).
- [ ] Lưu mọi prediction (`preds/{exp}/{fold}_{seed}.parquet`).
- [ ] Bảng hyperparameter đầy đủ + lệnh chạy.
- [ ] Báo cáo cả pooled-CV và LOCO; cả 3 ngưỡng (transferred/calibrated/oracle).
- [ ] Multi-seed CI (không chỉ fold-variance).

---

## 13. Lộ trình hiện thực (gợi ý thứ tự code)

1. `splitter.py` + `tests/test_splitter.py` (đảm bảo leakage-free trước tiên).
2. `segmentation.py` + `features.py` + cache (chạy `01_extract_features.py`).
3. `dataset.py` + `collate_bags` (kiểm tra shape bằng 1 batch nhỏ).
4. `encoders.py` → `transval_net.py` (forward chạy được với batch giả).
5. `grl.py` + `adversary.py` + `group_dro.py` (unit test gradient đảo dấu).
6. `trainer.py` E0→E1 (acoustic-only rồi fusion).
7. `calibration/` + `selective.py`.
8. `eval/` (metrics → confound_eval → stats → probe).
9. `cv_runner.py` ráp 5-fold × multi-seed.
10. `03_make_report.py` xuất EvaluationCard + bảng ablation.

---

### Ghi chú cuối
Mọi novelty của bài nằm ở **invariance + confound-control + calibration + thống kê đủ power**, KHÔNG ở fusion. Giữ fusion đơn giản có chủ đích để câu chuyện sạch: "chúng tôi đo tính hợp lệ chuyển giao được, và cho thấy điều gì thật sự sống sót". Đó là thứ khiến bài publishable bất kể kết quả dương hay âm.
