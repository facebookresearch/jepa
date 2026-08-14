# V-JEPA: Phân tích chuyên sâu & Kế hoạch trực quan hóa Manim (phong cách 3Blue1Brown)

> Tài liệu phân tích paper **"V-JEPA: Latent Video Prediction for Visual Representation Learning"** (ICLR 2024, anonymous submission) nhằm phục vụ việc xây dựng video hoạt họa giải thích mô hình theo phong cách 3Blue1Brown bằng Manim. Các thuật ngữ kỹ thuật được giữ nguyên bằng tiếng Anh để bảo vệ tính chính xác.

---

## PHẦN 1 — TỔNG QUAN & ĐỘNG LỰC NGHIÊN CỨU (Motivation & Problem Statement)

### 1.1. Bài toán V-JEPA giải quyết

V-JEPA (Video Joint-Embedding Predictive Architecture) giải quyết bài toán **self-supervised representation learning from video** —— học biểu diễn (representation) trực quan từ dữ liệu video mà **không cần nhãn (label) do con người gán**. Mục tiêu là xây dựng một **video foundation model** tổng quát: cùng một bộ mã hoá (encoder) đã pretrained có thể được dùng "off-the-shelf" cho nhiều downstream task (nhận dạng hành động, định vị hành động, phân loại ảnh tĩnh...) thông qua một lớp *frozen evaluation* (attentive probing) nhẹ, **không cần fine-tuning toàn bộ backbone**.

Trọng tâm cụ thể:
- Học các **feature cấp cao (semantic features)** thay vì chỉ học đặc trưng pixel/thấp cấp.
- Nắm bắt **motion dynamics** liên quan đến thời gian — thứ mà các ảnh tĩnh đơn lẻ không thể cung cấp.
- Đạt performance mạnh trong **frozen evaluation** (linear / attentive probe), thay vì phải fine-tune như các MAE-based models.

### 1.2. Hạn chế của các hướng tiếp cận trước đây

V-JEPA đặt mình đối lập với 2 dòng chính trong濾 self-supervised video learning:

**(A) Generative / Pixel-reconstruction models (MAE, VideoMAE, VideoMAEv2, OmniMAE):**
- Huấn luyện một encoder-decoder (decoder thường dùng để reconstruction) để **dự đoán lại raw pixels** của các video patch bị che (masked voxels).
- **Ưu điểm:** Inductive bias tối thiểu, mạnh khi fine-tuning toàn bộ mạng trên downstream task.
- **Hạn chế:**
  1. **Tốn tài nguyên cho decoder** mà không trực tiếp đóng góp vào feature quality.
  2. **Feature thu được kém trong frozen evaluation** — cần nhiều thủ tục adaptation phức tạp (Table 9 trong paper: VideoMAE chỉ 41.3% SSv2 với linear probe, so với V-JEPA là 50.1%).
  3. **Bắt buộc mô hình phải mất công dự đoán các chi tiết pixel không dự đoán được** (stochastic high-frequency texture, nhiễu, ánh sáng...) —— gây lãng phí năng lực học.
  4. Khó mở rộng vì chi phí pretraining lớn (VideoMAEv2 cần 1600M samples seen, gấp 6-7× so với V-JEPA chỉ 210M).

**(B) Weakly-supervised (CLIP, InternVideo, VATT):**
- Dựa vào text caption (như closed captions / ASR transcription) đồng hành cùng video.
- **Hạn chế:** Cần dữ liệu có caption (không thuần self-supervised), và vẫn kém performance SSD (motion-sensitive task SSv2) so với V-JEPA: VATT đạt 58.7, InternVideo 60.3, còn V-JEPA 71.2 —— chênh lệch tới +10/+11 points.

**(C) Image self-supervised (DINOv2, OpenCLIP, I-JEPA):**
- Mạnh ở xuất hiện (appearance) vì được pretrain trên dataset ảnh internet-scale (142M+ ảnh), nhưng **không nắm bắt được motion dynamics**: trên SSv2 DINOv2 chỉ 50.0, OpenCLIP chỉ 39.0, V-JEPA 71.2 (+21, +32 điểm). Các paper điểm qua *MC-JEPA*, *VITO*, *OmniMAE* cũng có một số hạn chế tương tự — đặc biệt của V-JEPA ám chỉ: tốt hơn frozen-eval, áp dụng masked-modelling online representation space, không phụ thuộc text.

### 1.3. Trực giác cốt lõi: vì sao dự đoán trong không gian tiềm ẩn (latent space) tốt hơn pixel level?

**Trực giác chính** được dẫn từ I-JEPA (Assran et al., 2023): trong video, phần lớn thông tin pixel-level **không thể dự đoán được** (stochastic, high-frequency, vượt khả năng predictive — ví dụ nhấp nháy hạt lấp lánh, leaves rustle, structure vô định của texture...). Khi ta buộc mô hình reconstruct pixel:
- Mô hình phải dành năng lực cho nhiễu không dự đoán được → giảm semantic richness.
- Mất cân bằng giữa "what is predictable" và "what is noisy".

**Giải pháp của JEPA** nói chung và V-JEPA nói riêng: thay vì reconstruct pixel, **dự đoán một biểu diễn trừu tượng (abstract latent)** của vùng masked. Vì latent representation đã được student/target encoder tự học để loại bỏ nhiễu, prediction trong không gian này:
- **Chỉ tập trung vào thông tin có thể dự đoán được** (predictable signal).
- Ép context encoder phải nắm bắt **semantic content** & **motion logic** chứ không phải texture.
- **Loại bỏ chi phí decoder** (học latent trực tiếp, không cần reconstruct).
- Quan trọng là **representation còn được học online trong quá trình training** (không phải freeze một CLIP space ngoài như Li et al., 2023) — → adapts để trở thành target tối ưu cho chính mình.

V-JEPA cho thấy trực giác này không chỉ hợp với ảnh (I-JEPA) mà còn **vượt trội hơn nữa với video** (vì video có nhiễu thời gian lớn hơn, redundancy cũng cao hơn → opportunity lớn hơn để cheat bằng temporal leakage nếu không thiết kế mask đúng cách).

---

## PHẦN 2 — KIẾN TRÚC & CƠ CHẾ KỸ THUẬT (Technical Deep Dive)

### 2.1. Khái quát: 3 mạng của JEPA

V-JEPA kế thừa ba mạng từ JEPA gốc:

| Thành phần | Vai trò | Backward gradient? |
|---|---|---|
| **Context Encoder** `E_θ(·)` | Mã hoá vùng *không bị che* (context) của video clip → tạo representation `z_N` | ✅ Có gradient |
| **Predictor** `P_φ(·,·)` | Từ `z_N` + mask tokens → dự đoán representation của vùng bị che | ✅ Có gradient |
| **Target Encoder** `Ẽ_θ(·)` | Mã hoá **toàn bộ** video clip (unmasked) → tạo representation target `s_L` | ❌ **Stop-gradient**, update bằng EMA |

> Thiết kế **non-symmetric**: predictor có gradient còn target thì không. Đây là chìa khoá chống collapse.

### 2.2. Luồng dữ liệu chi tiết (Data flow)

**Input video clip — $$16 \times 224 \times 224 \times 3$$**:
- 16 frames, temporal stride = 4, nên clip trải qua ~64 frames gốc = ~2 giây ở 30fps.
- Resized về 224×224 (hoặc 384 cho ViT-H/16₃₈₄).

**Tokenization (3D Conv):**
$$[16 \times 224 \times 224 \times 3] \xrightarrow{\text{3D Conv } d \text{ filters, kernel } 2 \times 16 \times 16, \text{stride } (2, 16, 16)} [8 \times 14 \times 14 \times d]$$

- Tubelet size = 2 frames, patch size = 16×16 pixels.
- Thêm **absolute 3D sin-cos positional embedding**.
- Flatten → sequence 1D có shape **$1568 \times d$** (với $8 \cdot 14 \cdot 14 = 1568$).

> Đối với ViT-L/16 $d = 1024$; ViT-H/16 $d = 1280$.

**Branch A — Context pathway (đường có gradient):**
$$ [L \times d] \xrightarrow{\text{remove masked patches (M tokens)}} [N \times d] \xrightarrow{E_\theta} [N \times d] $$$$ [N \times d] \xrightarrow{\text{concat learnable mask tokens } m_M \text{ (with 3D pos-emb)}} [L \times d] \xrightarrow{P_\phi} [M \times d] \quad (= \hat{s}_M) $$

**Branch B — Target pathway (đường stop-gradient):**
$$ [L \times d] \xrightarrow{\bar{E}_\theta} [L \times d] \xrightarrow{\text{remove unmasked patches}} [M \times d] \quad (= s_M) $$

**Loss** (giữa predicted và target, với stop-grad trên nhánh target):
$$ \mathcal{L} = \frac{1}{M} \sum_{k \in \{i_1,...,i_M\}} \big\| \hat{s}_k - s_k \big\|_1 $$

**Update rule:**
- Gradient backprop → cập nhật `θ` (context encoder) và `φ` (predictor) bằng AdamW.
- Target encoder weights `θ̄` được cập nhật bằng **EMA (Exponential Moving Average)** của context encoder weights (Polyak average):
$$ \bar{\theta}_{t+1} \leftarrow \tau \cdot \bar{\theta}_t + (1-\tau) \cdot \theta_{t+1} $$
với momentum `τ` khởi đầu 0.998, tăng dần tuyến tính lên 1.0 trong training.

### 2.3. Mask tokens & Predictor

- Mask tokens là một **shared learnable vector** + **3D sin-cos positional embedding** at vị trí masked.
- Predictor là **narrow ViT** (12 transformer blocks, embedding dim = 384) — nhẹ hơn context encoder rất nhiều để tránh việc predictor "làm hộ" context encoder (vì nếu predictor mạnh quá, nó có thể reconstruct target mà không cần context → collapse về một dạng trivial).
- Predictor nhận **cả z_N (context tokens) lẫn m_M (mask tokens)**, dùng joint space-time attention Соф để sinh $\hat{s}_M$.

### 2.4. Spatiotemporal Masking Strategy — **3D Multi-Block Masking**

Đây là một phần cực kỳ quan trọng của V-JEPA.

**Cấu trúc sản xuất mask:**
1. Sample **một số block không gian (spatial)** với kích thước/aspect ratio ngẫu nhiên.
2. **Lấy union** của các block này → mask 2D thời điểm đó.
3. **Lặp mask 2D này trên toàn bộ trục thời gian** (temporal ratio = 100%) — đây là tính chất "3D" — mask kéo dài xuyên suốt clip.

**Hai loại mask dùng song song (Multi-Mask):**

| Loại mask | So block | Spatial scale | Aspect ratio |
|---|---|---|---|
| **Short-range mask** | 8 | 0.15 | (0.75, 1.5) |
| **Long-range mask** | 2 | 0.7 | (0.75, 1.5) |

- Masking ratio trung bình ≈ **90%** → context encoder chỉ xử lý **~10%** video, đá__":
  + Tăng hiệu năng (efficient forward pass).
  + Ép task prediction khó → model buộc học semantic, không copy-through.
  + Ngừa temporal leakage: mask trải dài toàn bộ trục thời gian nên model không thể đơn giản "interpolate frame kế bên" để fill gap.

**Multi-mask prediction (amortize target compute):**
- Cùng một clip sample 2 mask khác nhau.
- Chạy context encoder + predictor **separately** cho mỗi mask.
- **Target encoder chỉ chạy 1 lần** duy nhất với full clip → dùng chung.
- Sinh 2 loss term → cộng lại (mỗi term có stop-grad target riêng).

**Ablation quan trọng (Appendix C.3):**
- Temporal coverage 100% + spatial 90% là tối ưu (Table 13: 0.50 acc).
- Temporal coverage 75%/50% gây trivial task → drop mạnh (0.10–0.16).
- Nhiều block nhỏ > ít block lớn (Table 14): 8 blocks 96×96 (0.50) > 1 block 192×192 (0.47).
- 2 mask/sample > 1 mask (Table 15: 0.55 vs 0.50).

### 2.5. Cơ chế chống Representation Collapse

**Vấn đề collapse:** Nếu cho cả target ↔ context cập nhật gradient hoặc không có gì can thiệp, nghiệm trivial: encoder output **hằng số** không phụ thuộc input → predictor học cũng output hằng số đó → loss = 0 mà không học gì.

**Cơ chế V-JEPA (kế thừa BYOL/EMA teacher):**
1. **Stop-gradient** trên target encoder → không có gradient backprop về target encoder thông qua loss.
2. **Target weights = EMA của context weights** (thay vì learnable riêng):
   - EMA là "moving average" → target **lùi hơn** context encoder về mặt evolution (chậm hơn).
   - Predictor (được huấn luyện trực tiếp) học **nhanh hơn** target encoder rất nhiều.
   - Do đó predictor luôn "theo sát" target encoder, có thể gần với **optimal predictor** (median) trong khi target encoder vẫn **chưa collapse**.
3. **Theoretical insight (Appendix D):** với $L_1$ loss, optimal predictor là **conditional median** của target; gradient của context encoder lúc đó = đẩy context encoder tới cung cấp đủ information để **giảm median absolute deviation (MAD)** của target conditioned trên context. Để gradient không trở thành zero (collapse), cần predictor optimal → khai thác target encoder slow-moving để đảm bảo predictor luôn "đuổi kịp" median → encode phải tiếp tục distribute đủ thông tin.

> Tóm lại: chống collapse bằng cách tạo khoảng cách thời gian giữa predictor (nhanh) và target encoder (chậm), ép encoder phải thật sự distribute thông tin tới predictor mới có thể minimize MAD.

---

## PHẦN 3 — KẾ HOẠCH TRỰC QUAN HOÁ MANIM (3Blue1Brown-Style Roadmap)

### 3.1. Danh sách khái niệm quan trọng nhất cần trực quan hoá

| # | Khái niệm | Độ ưu tiên |
|---|---|---|
| 1 | Video clip là một "khối 3D" (T×H×W) — gridding thành patches | ★★★ |
| 2 | Tokenizer 3D Conv: cube → flattened 1D sequence | ★★★ |
| 3 | 3D Multi-Block Masking: short-range và long-range, trải dài toàn trục thời gian | ★★★ |
| 4 | Ba-role architecture: Context Encoder / Predictor / Target Encoder với kết nối đặc biệt | ★★★ |
| 5 | EMA update & stop-gradient (chống collapse) | ★★ |
| 6 | $L_1$ loss giữa predicted vector và target vector | ★★ |
| 7 | Latent space so với pixel space — trực giác "predict semantics, not pixel" | ★★★ |
| 8 | Multi-mask prediction (sử dụng chung target) | ★ |
| 9 | Comparison với VideoMAE (decoder = pixel reconstruction) | ★★ |

### 3.2. Cấu trúc câu chuyện (Storyline)

Đề xuất chia video thành **6 chương** — mỗi chương kể một ý, dẫn dắt mạch lạc kiểu 3Blue1Brown:

**Chương 0 — Hook / Intro (~30s)**
- Một câu hỏi trực diện: *"Làm sao máy tính học hiểu được video — nơi mà đa số thông tin pixel là nhiễu không dự đoán được?"*
- Show một chuỗi frame (đơn giản) và zoom từ photorealistic → biễu tượng cube.
- Prommise câu trả lời: V-JEPA — جهان trí tuệ học bằng cách **điền chỗ trống** (fill in the blank) nhưng ở "world model" trừu tượng.

**Chương 1 — Trực giác (Intuition) (~1m30s)**
- So sánh 2 стратегии:
  + Pixel reconstruction (VideoMAE): ép model lấp đầy texture nhiễu → lãng phí năng lực.
  + Latent prediction (V-JEPA): chỉ predict cái gì có thể predict → học semantic.
- Hình tượng: một hình khối ramdom noise → collapse về cụm abstraction (播放\
▮ opts giảm entropy), trong khi V-JEPA chỉ encode phần "predictable signal" (có thể dùng mũi tên "Đoán được" vs "Không đoán được").

**Chương 2 — Từ video clip sang tokens (~1m)**
- Video clip = khối hình hộp chữ nhật (T × H × W × 3).
- Áp dụng **3D Conv với kernel 2×16×16** → chia thành các "tubelet" có dạng cube nhỏ 8×14×14 = 1568 cube.
- Flatten thành trục số (number line) các vector embedding → gắn 3D positional sin-cos.

**Chương 3 — Masking Strategy (~1m30s)**
- Show mặt cắt (T × H × W nhìn từ góc nghiêng).
- Generate các block hình chữ nhật ngẫu nhiên trên mặt (H×W) → union → mask 2D.
- **Repeat-to-extrude** theo trục T → mask 3D trải dài toàn bộ clip.
- Hiển thị 2 variant: short-range (8 blocks nhỏ) + long-range (2 blocks lớn).
- Cao trào: ~90% bị "xoá" — chỉ còn ~10% lấp lánh. Cảnh: model nhận "10% gợi ý" và phải đoán phần còn lại.

**Chương 4 — Architecture & Data Flow (~2m30s)**
- Đặt vào layout 3 mạng:
  - **Context Encoder** (ViT-L/H) phía trái — nhận ~10% patches (visible) → output `z_N`.
  - **Predictor** (narrow ViT-12, dim 384) ở giữa — nhận `z_N` + Mask Tokens `m_M` → output `ŝ_M`.
  - **Target Encoder** (ViT-L/H, identical init) phía phải — nhận **FULL** clip → output `s_L` → giữ lại `M` masked patches → `s_M`.
- Animation info flow:
  + Balls/chunk di chuyển từ input → context encoder → predictors → output predicted vectors (pink).
  + Balls riêng lặp từ full input → target encoder → output target vectors (blue) với **stop-gradient** biểu diễn bằng "mìmi wall ⛔".
- So sánh `ŝ_k` và `s_k` với $L_1$ distance line giữa từng cặp vector → average loss.
- Cuối: arrow cập nhật gradient về Context Encoder & Predictor (mũi tên đặc), arrow EMA từ Context Encoder tới Target Encoder (dashed/slow arrow, biểu diễn momentum).

**Chương 5 — Collapse Prevention (~1m30s)**
- Concept: "What if encoder cheated?" — Show một kịch bản tệ: encoder output một constant vector bất chấp input → loss = 0.
- Sau đó show cơ chế chống:
  + Stop-gradient ngăn backprop làm target encoder "easy collapse".
  + EMA khiến target encoder **di chuyển chậm hơn** predictor, predictor "đuổi kịp" → encoder buộc phân to thông tin để predictor vẫn có reasons để đoán.
- Sử dụng analogy "target encoder là một tấm bia chậm trễ" — predictor mở ra "blueprint" của bia, encoder phải vẽ đủ chi tiết để bia ở đằng sau cũng khớp.

**Chương 6 — Results & Insight (~1m)**
- Bar chart so sánh:
  + SSv2: V-JEPA 71.2 vs VideoMAE 61.2 (+10) vs DINOv2 50.0 (+21) vs OpenCLIP 39.0 (+32)
  + K400: 82.1 (+4 so với VideoMAE)
  + Sample efficiency: V-JEPA 210M samples vs VideoMAEv2 1600M, OpenCLIP 39000M.
- Final: brief message — *"latent prediction + 3D masking + EMA teacher = efficient, general, semantic video foundation."*

### 3.3. Gợi ý Manim cụ thể cho mỗi concept

| Khái niệm | Cấu trúc Manim đề xuất |
|---|---|
| **Video clip 3D** | Dùng `ThreeDScene` + `VGroup` của các `Cube` (frames xếp chồng). Apply `ApplyMethod(camera.move_to)` để quay quanh. |
| **3D Conv tokenization** | `Surface` cho video clip rồi `SurroundingRectangle` khu vực tubelet + `Transform` thành `Dot` vector trên trục. |
| **3D Multi-Block Mask** | `ThreeDScene`, `VGroup(Cube)` cho patches; patches masked → `FadeToColor(BLACK)`/`SetOpacity(0.3)` + `Cube` biến mất. Repeat-to-extrude = `AnimationGroup` của `ApplyMethod` stack patches dọc theo timestamp. |
| **Predicted/Target vector** | `Matrix` object hoặc `Vector` (column) với các entry được highlight từng cặp bằng `Arrow` giữa chúng (`L_1` distance). |
| **Encoder ↔ Predictor ↔ Target info flow** | `Rectangle` (label) + `Arrow` nối, dùng `ShowPassingFlash` cho "ball flow". Stop-gradient = `Cross`/`Line` trên arrow. EMA = `DashedVMobject` arrow pulse slow. |
| **Loss discrepancy** | `Brace` + `Tex("$\hat{s}_k - s_k$")` + `ValueTracker` cho $L_1$ distance shrinking. |
| **EMA animation** | `ValueTracker` thẳng từ 0 → 1 cho momentum, `Become` từ dashed → solid arrow; tăng độ "đậm" dần khi momentum tăng dần tới 1.0. |
| **Pixel vs Latent intuition** | `ImageMobject` (random noise pattern) bên trái → một loạt `BarChart` của entropy lớn; bên phải `Vector` trừu tượng phẳng hơn — chỉ còn biểu tượngsemantic. Dùng `Transform` từ noise sang smooth latent cube. |
| **Sample efficiency bar** | `BarChart` với trục log (dùng `NumberLine`) → V-JEPA ngắn 1-2 orders of magnitude. |
| **Collapse prevention** | Show một `Vector` constant (một điểm không di chuyển dù input thay đổi); circle màu đỏ; sau đó `Indicate` cờ và "tháo gỡ" bằng EMA + stop-grad → vector bắt đầu boust varied. |

### 3.4. Phụ chú về phong cách 3B1B

- **Màu sắc:** Thống nhất:
  - **Context branch** = xanh dương (#5DADE2)
  - **Predictor branch** = hồng/cam (#E59866)
  - **Target branch** = xanh lục (#58D68D)
  - **Masked region** = xám (0.3 opacity)
  - **Loss/grad arrows** = vàng (#F4D03F)
- **Pacing:** Mỗi concept cần ~15-25 giây cho phép khán giả absorb; tránh nhồi quá nhiều text.
- **Toán học inline:** Dùng `MathTex` cho các công thức như Loss $L_1$, EMA update rule, conditional median.
- **Câu dẫn dắt:** Nên có 1 texture giọng "narrator" with hooks như *"But there's a subtle trap..."* → collapsse, *"Here's the clever part."* → EMA.
- **Trigger visuals:** Khi nhắc "gradient" → đèn vàng flow; khi nhắc "no gradient" → đèn xanh但有 wall.

---

## PHẦN 4 — TÓM TẮT 3 Ý CỐT LÕI CỦA V-JEPA

1. **Latent prediction thay cho pixel reconstruction:** V-JEPA dự đoán biểu diễn trừu tượng của vùng masked trong một latent space được học online, loại bỏ nhiễu không predict được → học semantic feature mạnh hơn nhiều so với VideoMAE khi đánh giá frozen.

2. **3D Multi-Block Masking:** Mask block không gian trải dài **toàn bộ trục thời gian** (~90% masked), chỉ cho encoder xử lý ~10% video → vừa efficient vừa ép task đủ khó để tránh temporal leakage, đồng thời multi-mask prediction chia sẻ target để tiết kiệm compute.

3. **Chống collapse bằng EMA + stop-gradient:** Target encoder **không** nhận gradient từ loss; chỉ updated bằng EMA của context encoder, đảm bảo predictor (nhanh) luôn đuổi kịp target (chậm) → encoder buộc distribute thông tin thay vì collapse về hằng số. Đây là cơ chế lý thuyết (median/MAD) nhất quán và đã được kiểm chứng experimentally.

---

## PHỤ LỤC — THAM CHIẾU SỐ LIỆU QUICK-LOOK

| Metric | V-JEPA (ViT-H/16₃₈₄, VideoMix2M) | Best baseline | Δ |
|---|---|---|---|
| K400 (frozen) | **82.1** | VideoMAE 77.9 | +4.2 |
| SSv2 (frozen) | **71.2** | VideoMAE 61.2 | +10.0 |
| AVA (frozen, mAP) | **25.8** | VideoMAE 21.6 | +4.2 (best 25.0) |
| IN1K (frozen, 2-layer) | 77.9 | DINOv2 86.2 | -8.3 (về image task, image-pretrain vẫn mạnh hơn) |
| Samples seen during pretraining | **210M** | OpenCLIP 39000M | 2 orders of magnitude ít hơn |

> Ghi chú: V-JEPA vẫn kém image-pretrain model trên image downstream thuần (IN1K, Places, iNat21) — giới hạn chính là độ đa dạng (diversity) của video dataset so với internet-scale image corpus. Hướng future: cải thiện video data scale/diversity.

---

*Taì liệu này phục vụ làm spine cho việc viết script chi tiết và code Manim từng chương — có thể chia thành 6 file scene riêng (`intro.py`, `intuition.py`, `tokenizer.py`, `masking.py`, `architecture.py`, `results.py`).*