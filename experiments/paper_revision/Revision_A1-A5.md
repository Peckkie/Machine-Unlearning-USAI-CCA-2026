# Paper revision: items [5]–[8] (A2, A1, A3, A5): แก้จาก PDF `_RadiologyAI__Machine_Unlearning.pdf`

> ทุกตัวเลขคำนวณใหม่จาก **prediction / distance เดิมที่ใช้ทำตารางใน paper** (ไม่ได้เทรนใหม่) และตรวจแล้วว่าค่าที่ตรงกับตารางเดิมตรงทุกตัว
> สคริปต์: `experiments/analysis/a2_stats.py`, `a1_face_metrics.py`, `a3_flip_consistency.py` · ผลดิบ: `/media/tohn/HDD2/Model_unlearn_2026/analysis/` (เครื่อง 29) และ `/media/HDD/Model_unlearn_2026/analysis/` (เครื่อง 28)
> ข้อความใหม่ไม่มี em-dash · ตารางใหม่ตั้งชื่อชั่วคราวเป็น Table A / B / C ให้เลขใหม่ตอนรวมไฟล์ (Table 5 เดิม = manipulation check)

---

## [5] A2: Ultrasound: case-level bootstrap CI, McNemar, balanced accuracy, per-class (กรรมการข้อ 6, 8)

### ผลที่ได้ (ไว้อ้างอิง)

| Backbone | Test | Metric | Original | Unlearned | Δ (95% CI) |
|---|---|---|---|---|---|
| EfficientNet-B5 | Lab (1,312 img / 144 cases) | Accuracy | 0.835 [0.785, 0.885] | 0.886 [0.844, 0.929] | **+0.051 [+0.032, +0.071]** |
| | | Macro-F1 | 0.626 [0.579, 0.663] | 0.755 [0.708, 0.799] | **+0.130 [+0.098, +0.167]** |
| | | Balanced acc. | 0.572 [0.530, 0.628] | 0.710 [0.661, 0.774] | **+0.138 [+0.102, +0.178]** |
| | | McNemar (exact) | 23 only-orig correct | 90 only-unl correct | **p = 1.5 × 10⁻¹⁰** |
| EfficientNet-B5 | Field (807 img / 51 cases) | Accuracy | 0.648 [0.577, 0.719] | 0.849 [0.797, 0.892] | **+0.201 [+0.151, +0.253]** |
| | | Macro-F1 | 0.165 [0.083, 0.227] | 0.174 [0.094, 0.239] | +0.010 [−0.037, +0.050] |
| | | Balanced acc. | 0.157 [0.091, 0.228] | 0.270 [0.198, 0.389] | **+0.113 [+0.042, +0.193]** |
| | | Abnormal sens. / spec. | 0.645 / 0.694 | 0.452 / 0.899 | |
| | | McNemar | 12 | 174 | **p = 5.5 × 10⁻³⁸** |
| ResNet-152V2 | Lab | Accuracy | 0.857 [0.814, 0.899] | 0.864 [0.818, 0.909] | +0.008 [−0.008, +0.024] |
| | | Macro-F1 | 0.662 [0.598, 0.714] | 0.711 [0.644, 0.762] | +0.048 [0.000, +0.100] |
| | | Balanced acc. | 0.631 [0.571, 0.699] | 0.672 [0.610, 0.743] | +0.041 [−0.008, +0.095] |
| | | McNemar | 52 | 62 | p = 0.40 |
| ResNet-152V2 | Field | Accuracy | 0.786 [0.732, 0.834] | 0.880 [0.832, 0.920] | **+0.094 [+0.067, +0.122]** |
| | | Macro-F1 | 0.199 [0.107, 0.283] | 0.289 [0.149, 0.351] | **+0.090 [+0.016, +0.121]** |
| | | Balanced acc. | 0.401 [0.254, 0.520] | 0.381 [0.274, 0.500] | −0.020 [−0.116, +0.102] |
| | | Abnormal sens. / spec. | 0.565 / 0.827 | 0.468 / 0.928 | |
| | | McNemar | 17 | 93 | **p = 7.2 × 10⁻¹⁴** |
| ViT-L/32 | Field | Accuracy | 0.731 [0.673, 0.782] | 0.760 [0.702, 0.809] | +0.029 [−0.007, +0.062] |
| | | Macro-F1 | 0.179 [0.087, 0.230] | 0.145 [0.081, 0.183] | −0.034 [−0.078, +0.027] |
| | | McNemar | 67 | 90 | p = 0.079 |

**สิ่งที่ต้องรู้ก่อนเขียน (สำคัญ)**
1. EfficientNet-B5 lab: ดีขึ้นจริงทุก metric, CI ของ Δ ไม่คร่อม 0, McNemar p < 10⁻⁹ → ใช้คำว่า "significant" ได้
2. **ResNet-152V2 lab: ไม่ significant** (Δ acc CI คร่อม 0, McNemar p = 0.40) → paper เดิมที่บอกว่า ResNet ดีขึ้นบน lab ต้องลดน้ำเสียง
3. **Field test: accuracy ที่เพิ่มมาจากคลาส Normal เป็นหลัก**, sensitivity ของ Abnormal **ลดลง** ทั้ง EfficientNet-B5 (0.645 → 0.452) และ ResNet-152V2 (0.565 → 0.468) ขณะที่ specificity เพิ่ม; macro-F1 ของ EfficientNet-B5 ไม่ต่าง (CI คร่อม 0) → ต้องเขียนตรงไปตรงมา ไม่งั้นกรรมการจะจับได้
4. ViT: ไม่มีอะไร significant (สอดคล้องกับที่ paper บอกว่า ViT ไม่ได้ประโยชน์)
5. Per-class (EfficientNet-B5, lab): AB11 sensitivity 0.182 → 0.909 (+40 จาก 55 ภาพ = Renal Cyst/Stone ที่ paper พูดถึง ✓), AB10 0.40 → 0.80, AB083 0.455 → 0.818; ลดลง: AB06 0.619 → 0.524, AB04 0.632 → 0.605

### [5.1] Section 4.2: วิธีวิเคราะห์สถิติ

**เดิม**
> To ensure the reproducibility of our single-run experiments, the dataset splits were explicitly fixed into distinct folders, and all model initializations and training configurations were controlled using a predefined random seed. All precision, recall, and F1 values are macro-averaged across the 15 classes.

**ใหม่**
> The dataset splits were explicitly fixed into distinct folders. Precision, recall, and F1 are macro-averaged over the classes that occur in either the ground truth or the predictions of each test set (the scikit-learn convention); because the field test set contains only 10 of the 15 classes, we additionally report balanced accuracy (the mean per-class recall over the classes present in the ground truth) and the sensitivity and specificity of the Normal versus Abnormal decision. Uncertainty is quantified with a case-level cluster bootstrap (2,000 resamples of patients, 144 cases in the laboratory test set and 51 in the field test set), which respects the dependence between images of the same patient; for each comparison we report the 95% percentile interval of the paired difference (unlearned minus original) computed on the same resamples. Paired per-image differences in correctness are tested with an exact McNemar test, and per-class sensitivity and specificity are reported with Clopper–Pearson 95% intervals (Supplementary Table S[x]).

> ⚠️ ประโยคเดิม "controlled using a predefined random seed" **ไม่ตรงกับโค้ด** (โค้ดเดิมไม่ได้ตั้ง seed) จึงตัดออก; ถ้าทำ X2 (3 seeds) เสร็จ ให้ใช้ประโยคจาก planner E11 แทน

### [5.2] Section 4.3: เพิ่มตารางใหม่ (Table A) หลัง Table 4

**ใหม่ (เพิ่ม)**
> **Table A:** Paired comparison of original and unlearned models with case-level bootstrap 95% confidence intervals (2,000 resamples of patients) and exact McNemar tests on per-image correctness. Δ = unlearned minus original. Bold: interval excludes zero.
>
> | Backbone | Test set | Metric | Original | Unlearned | Δ [95% CI] | McNemar p |
> |---|---|---|---|---|---|---|
> | EfficientNet-B5 | Lab (n = 1,312; 144 cases) | Accuracy | 0.835 [0.785, 0.885] | 0.886 [0.844, 0.929] | **+0.051 [0.032, 0.071]** | 1.5 × 10⁻¹⁰ |
> | | | Macro-F1 | 0.626 [0.579, 0.663] | 0.755 [0.708, 0.799] | **+0.130 [0.098, 0.167]** | |
> | | | Balanced accuracy | 0.572 [0.530, 0.628] | 0.710 [0.661, 0.774] | **+0.138 [0.102, 0.178]** | |
> | | Field (n = 807; 51 cases) | Accuracy | 0.648 [0.577, 0.719] | 0.849 [0.797, 0.892] | **+0.201 [0.151, 0.253]** | 5.5 × 10⁻³⁸ |
> | | | Macro-F1 | 0.165 [0.083, 0.227] | 0.174 [0.094, 0.239] | +0.010 [−0.037, 0.050] | |
> | | | Balanced accuracy | 0.157 [0.091, 0.228] | 0.270 [0.198, 0.389] | **+0.113 [0.042, 0.193]** | |
> | ResNet-152V2 | Lab | Accuracy | 0.857 [0.814, 0.899] | 0.864 [0.818, 0.909] | +0.008 [−0.008, 0.024] | 0.40 |
> | | | Macro-F1 | 0.662 [0.598, 0.714] | 0.711 [0.644, 0.762] | +0.048 [0.000, 0.100] | |
> | | | Balanced accuracy | 0.631 [0.571, 0.699] | 0.672 [0.610, 0.743] | +0.041 [−0.008, 0.095] | |
> | | Field | Accuracy | 0.786 [0.732, 0.834] | 0.880 [0.832, 0.920] | **+0.094 [0.067, 0.122]** | 7.2 × 10⁻¹⁴ |
> | | | Macro-F1 | 0.199 [0.107, 0.283] | 0.289 [0.149, 0.351] | **+0.090 [0.016, 0.121]** | |
> | | | Balanced accuracy | 0.401 [0.254, 0.520] | 0.381 [0.274, 0.500] | −0.020 [−0.116, 0.102] | |
> | ViT-L/32 | Field | Accuracy | 0.731 [0.673, 0.782] | 0.760 [0.702, 0.809] | +0.029 [−0.007, 0.062] | 0.079 |
> | | | Macro-F1 | 0.179 [0.087, 0.230] | 0.145 [0.081, 0.183] | −0.034 [−0.078, 0.027] | |

### [5.3] Section 4.3: ย่อหน้าผล EfficientNet-B5 lab (เพิ่มท้ายย่อหน้า)

**เดิม**
> ...demonstrating that flip-invariance unlearning not only addresses symmetry bias but also enhances overall classification performance beyond specialized architectures.

**ใหม่**
> ...demonstrating that flip-invariance unlearning not only addresses symmetry bias but also enhances overall classification performance beyond specialized architectures. The gain is not attributable to sampling variation in the test set: under the case-level bootstrap, the improvement in accuracy is +0.051 (95% CI 0.032 to 0.071), in macro-F1 +0.130 (0.098 to 0.167), and in balanced accuracy +0.138 (0.102 to 0.178), and the unlearned model corrects 90 images that the original model misclassifies while introducing 23 new errors (exact McNemar p = 1.5 × 10⁻¹⁰; Table A). The largest per-class gain is in the renal cyst and stone class (AB11), whose sensitivity rises from 0.18 to 0.91 (40 of 55 images), consistent with the left kidney versus spleen confusion analyzed in Section 4.1.

### [5.4] Section 4.3: ResNet-152V2 lab (ลดน้ำเสียง)

**เดิม** (ย่อหน้า "To contextualize our results, ..." ไม่มีการพูดถึง ResNet lab แยก, **เพิ่มประโยคใหม่หลังย่อหน้า BiTNet**)

**ใหม่ (เพิ่ม)**
> For ResNet-152V2, the laboratory-set differences are small and not statistically reliable: accuracy changes by +0.008 (95% CI −0.008 to 0.024), balanced accuracy by +0.041 (−0.008 to 0.095), and the paired McNemar test is not significant (p = 0.40). We therefore do not claim a laboratory-set improvement for ResNet-152V2; its benefit appears on the field test set (below).

### [5.5] Section 4.3: ย่อหน้า field test

**เดิม**
> The experimental results from Table 4 (field testset) clearly demonstrate significant differences performance improvement when employing the unlearning strategies under real-world conditions. For Table 4, which tested on the field testset comprising 807 images from 51 cases collected during field visits, ResNet-152V2 continued to show the best performance (accuracy 0.88, f1-score 0.29), while EfficientNet-B5 suffered from low precision and recall, despite some improvement in accuracy.

**ใหม่**
> On the field test set (807 images from 51 cases collected during field visits; Table 4), both CNNs improve in accuracy after unlearning: EfficientNet-B5 by +0.201 (95% CI 0.151 to 0.253; McNemar p = 5.5 × 10⁻³⁸) and ResNet-152V2 by +0.094 (0.067 to 0.122; p = 7.2 × 10⁻¹⁴), with ResNet-152V2 also improving in macro-F1 (+0.090, 0.016 to 0.121; Table A). These gains must be read together with the strong class imbalance of this set (745 of 807 images are Normal): for both CNNs the Normal-versus-Abnormal specificity increases (EfficientNet-B5 0.69 to 0.90; ResNet-152V2 0.83 to 0.93) while the sensitivity for abnormal images decreases (0.65 to 0.45 and 0.57 to 0.47, respectively), and the macro-F1 of EfficientNet-B5 does not change reliably (+0.010, −0.037 to 0.050). The field-set accuracy gain therefore reflects mainly fewer false-positive abnormal calls on normal scans rather than better detection of abnormalities, and we interpret it as a robustness result under distribution shift, not as improved abnormality detection.

### [5.6] Section 4.3: ย่อหน้า confusion matrix (ลดคำว่า "significantly")

**เดิม**
> These results demonstrate that CNN-based models, which inherently learn symmetrical concepts through flip augmentation during pre-training, benefit significantly from our unlearning approach. The process effectively removes harmful biases while preserving beneficial features learned from the original dataset.

**ใหม่**
> These results indicate that CNN-based models pre-trained with flip augmentation can benefit from the unlearning stage; for EfficientNet-B5 the benefit is statistically reliable on the laboratory test set (Table A). We do not claim that the procedure removes the bias completely; Section 4.4 [A3] quantifies how much flip invariance the representations retain.

---

## [6] A1: Face matching: ROC-AUC, EER, TAR@FAR (กรรมการข้อ 11, 12, 13)

### ผลที่ได้

| Metric | Baseline | Unlearned | Δ (95% CI) |
|---|---|---|---|
| ROC-AUC (= CLES) | 0.5535 [0.5186, 0.5895] | 0.7663 [0.7224, 0.8067] | **+0.213 [+0.178, +0.246]** |
| EER | 0.461 [0.435, 0.488] | 0.286 [0.250, 0.325] | **−0.175 [−0.204, −0.143]** |
| TAR @ FAR = 1% | 0.021 [0.012, 0.033] | 0.088 [0.063, 0.116] | **+0.066 [+0.045, +0.095]** |
| TAR @ FAR = 0.1% | 0.005 [0.001, 0.011] | 0.019 [0.010, 0.031] | **+0.014 [+0.005, +0.027]** |

identity-level cluster bootstrap 2,000 resamples ของ 30 identities · 8,339 same / 75,808 different pairs (ไฟล์ระยะเดียวกับ Table 6) · ROC-AUC = CLES ใน Table 6 ตรงกัน ✓

> ⚠️ ค่า TAR@FAR ต่ำมากในเชิงสัมบูรณ์ (9% ที่ FAR 1%) → ต้องเขียนว่า unlearning ทำให้ embedding แยก identity ได้ดีขึ้น **แต่ยังไม่ใช่ face verifier ที่ใช้งานจริงได้**

### [6.1] Section 5.4: เพิ่ม subsection ใหม่หลัง RQ2

**ใหม่ (เพิ่ม)**
> **Verification metrics.** To express the separation in the standard face-verification form, we treat each pair as a verification trial in which the two images are declared to be the same person when their embedding distance falls below a threshold. Table B reports the area under the ROC curve, the equal error rate (EER), and the true-accept rate at false-accept rates of 1% and 0.1%, with 95% confidence intervals from an identity-level cluster bootstrap (2,000 resamples of the 30 test identities). The ROC-AUC equals P(D_diff > D_same) and therefore coincides with the CLES in Table 6. Unlearning raises the ROC-AUC from 0.554 to 0.766 (Δ = +0.213, 95% CI 0.178 to 0.246), lowers the EER from 0.461 to 0.286 (Δ = −0.175, −0.204 to −0.143), and increases the true-accept rate at 1% false accepts from 0.021 to 0.088 (Δ = +0.066, 0.045 to 0.095). In absolute terms these operating points remain far below those of dedicated face-recognition systems, which is expected for an ImageNet backbone fine-tuned with a single triplet-loss configuration on 30 identities; the experiment is intended to show the direction and size of the change in embedding geometry, not a deployable verifier.
>
> **Table B:** Face-verification metrics computed from the same-person (n = 8,339) and different-person (n = 75,808) distances of Table 6. 95% CI from an identity-level cluster bootstrap (2,000 resamples of 30 identities).
>
> | Metric | Baseline | Unlearned | Δ [95% CI] |
> |---|---|---|---|
> | ROC-AUC | 0.554 [0.519, 0.590] | 0.766 [0.722, 0.807] | +0.213 [0.178, 0.246] |
> | EER (lower is better) | 0.461 [0.435, 0.488] | 0.286 [0.250, 0.325] | −0.175 [−0.204, −0.143] |
> | TAR @ FAR = 1% | 0.021 [0.012, 0.033] | 0.088 [0.063, 0.116] | +0.066 [0.045, 0.095] |
> | TAR @ FAR = 0.1% | 0.005 [0.001, 0.011] | 0.019 [0.010, 0.031] | +0.014 [0.005, 0.027] |

### [6.2] Section 6.3 Limitations: ประโยคที่ตอนนี้ทำแล้ว

**เดิม**
> Although the statistical significance of our findings is unambiguous given the large sample size, the practical impact on downstream tasks such as face verification accuracy should be quantified through standard metrics such as ROC-AUC and equal-error rate.

**ใหม่**
> The large number of pairs makes nominal p-values very small, but the independent units are the 30 test identities; we therefore base all intervals on an identity-level bootstrap and report standard verification metrics (Table B). Although unlearning improves every verification metric, the absolute true-accept rates remain low (0.088 at a 1% false-accept rate), so the face experiment demonstrates a change in embedding geometry rather than a practical face-recognition system.

### [6.3] Abstract: ประโยค face matching (เพิ่มตัวเลขมาตรฐาน)

**เดิม**
> In a face-matching evaluation on 8,339 same-person and 75,808 different-person pairs, flip-invariance unlearning raised the probability that a different-person embedding lies farther apart than a same-person embedding from CLES = 0.55 to 0.77 (Cliff's δ from 0.11 to 0.53).

**ใหม่**
> In a face-matching evaluation on 8,339 same-person and 75,808 different-person pairs from 30 held-out identities, the verification ROC-AUC rose from 0.55 to 0.77 and the equal error rate fell from 0.46 to 0.29.

---

## [7] A3: Flip-consistency score (กรรมการข้อ 1, 14, 15)

### ผลที่ได้ (cosine similarity ระหว่าง feature ของภาพกับภาพที่ flip ซ้ายขวา; สูง = invariant ต่อ flip มาก)

| Model | Lab (n = 1,312) | Field (n = 807) |
|---|---|---|
| EfficientNet-B5 backbone, ImageNet | 0.981 ± 0.008 | 0.984 ± 0.007 |
| EfficientNet-B5 backbone, after unlearning (B4-B7) | 0.942 ± 0.028 | 0.972 ± 0.010 |
| EfficientNet-B5 original (Table 3) | 0.925 ± 0.056 | 0.960 ± 0.025 |
| EfficientNet-B5 unlearned (Table 3) | 0.904 ± 0.061 | 0.936 ± 0.033 |
| ResNet-152V2 original (Table 3) | 0.728 ± 0.139 | 0.762 ± 0.093 |
| ResNet-152V2 unlearned (Table 3) | 0.662 ± 0.222 | 0.710 ± 0.146 |

| Comparison (unlearned − reference) | Lab Δ [case-level 95% CI] · % images lower · Wilcoxon p | Field Δ [95% CI] · % lower · p |
|---|---|---|
| Backbone: unlearned vs ImageNet | **−0.039 [−0.042, −0.037]** · 97% · 6 × 10⁻²¹⁴ | **−0.012 [−0.014, −0.011]** · 91% · 1 × 10⁻¹¹⁵ |
| EfficientNet-B5 downstream | **−0.020 [−0.023, −0.017]** · 73% · 2 × 10⁻⁶⁵ | **−0.024 [−0.026, −0.022]** · 88% · 5 × 10⁻¹⁰² |
| ResNet-152V2 downstream | **−0.067 [−0.078, −0.056]** · 59% · 2 × 10⁻²⁹ | **−0.052 [−0.067, −0.038]** · 61% · 3 × 10⁻¹⁹ |

**สิ่งที่ต้องรู้ก่อนเขียน**
1. ทิศทางตรงกันทุกคู่ ทั้ง lab และ field: หลัง unlearning ค่า cosine **ลดลง** (invariant ต่อ flip น้อยลง) และ CI ไม่คร่อม 0 → เป็นหลักฐานระดับ representation ที่กรรมการขอ
2. **แต่ลดลงไม่มาก**: backbone ยังมี cosine 0.94 (จาก 0.98) → representation ยังเกือบ invariant ต่อ flip; สนับสนุนชื่อเรื่องใหม่ "Reducing inherited flip invariance" และต้องไม่อ้างว่า "removes"
3. การเปรียบเทียบที่สะอาดที่สุดคือ **backbone ก่อน downstream** (ImageNet vs หลัง unlearn) เพราะยังไม่มีผลของ downstream fine-tuning มาปน
4. Prediction agreement (argmax เดิม = argmax หลัง flip) **ไม่ใช้เป็นผลหลัก**: บน field test agreement เพิ่ม (EfficientNet-B5 0.84 → 0.91) เพราะโมเดลทาย Normal บ่อยขึ้น (ดู [5]) จึงตีความยาก ให้ใส่ใน supplementary เท่านั้น
5. ยังไม่ได้วัด ViT-L/32 (ถ้ากรรมการถาม ทำเพิ่มได้ด้วยสคริปต์เดียวกัน)

### [7.1] Section 4.4 ใหม่ (เพิ่มหลังย่อหน้า confusion matrix ของ 4.3, หน้า 16 ก่อน Section 5)

**ใหม่ (เพิ่ม)**
> **4.4 Representation-level evidence: flip consistency**
>
> Downstream accuracy shows that the unlearning stage helps, but not that it changes the property it targets. We therefore measured flip consistency directly: for each test image x we computed the cosine similarity between the penultimate (global-average-pooled) features of x and of its exact horizontal mirror, s(x) = cos(f(x), f(flip(x))); a value of 1 means the representation is perfectly invariant to the flip. We evaluated the EfficientNet-B5 backbone before and after the unlearning stage (both applied directly to ultrasound images, before any ultrasound training) and the fine-tuned original and unlearned models of Table 3, using forward passes only and the same preprocessing as in training. Differences were tested with a paired Wilcoxon signed-rank test, and 95% intervals of the mean paired difference were obtained with the case-level bootstrap of Section 4.2.
>
> Table C shows that the unlearning stage lowers flip consistency in every comparison and on both test sets. On the laboratory set the backbone similarity falls from 0.981 to 0.942 (mean paired difference −0.039, 95% CI −0.042 to −0.037; lower for 97% of images; Wilcoxon p < 10⁻¹⁰⁰), and the reduction persists after ultrasound fine-tuning for both EfficientNet-B5 (−0.020, −0.023 to −0.017) and ResNet-152V2 (−0.067, −0.078 to −0.056); the field set shows the same pattern. The effect is, however, partial: after unlearning the EfficientNet-B5 backbone still maps an image and its mirror to features with a cosine similarity of 0.94 on average. The auxiliary task therefore reduces, rather than eliminates, the flip invariance inherited from pre-training, which is how we describe the method throughout the paper.
>
> **Table C:** Flip consistency, mean ± SD of cos(f(x), f(flip(x))) over test images (higher = more flip-invariant), and the mean paired difference (unlearned minus reference) with case-level bootstrap 95% CI. All differences: Wilcoxon p < 10⁻¹⁸.
>
> | Model | Lab (n = 1,312) | Field (n = 807) |
> |---|---|---|
> | EfficientNet-B5 backbone, ImageNet weights | 0.981 ± 0.008 | 0.984 ± 0.007 |
> | EfficientNet-B5 backbone, after unlearning | 0.942 ± 0.028 | 0.972 ± 0.010 |
> | Δ backbone | −0.039 [−0.042, −0.037] | −0.012 [−0.014, −0.011] |
> | EfficientNet-B5, original (Table 3) | 0.925 ± 0.056 | 0.960 ± 0.025 |
> | EfficientNet-B5, unlearned (Table 3) | 0.904 ± 0.061 | 0.936 ± 0.033 |
> | Δ EfficientNet-B5 | −0.020 [−0.023, −0.017] | −0.024 [−0.026, −0.022] |
> | ResNet-152V2, original (Table 3) | 0.728 ± 0.139 | 0.762 ± 0.093 |
> | ResNet-152V2, unlearned (Table 3) | 0.662 ± 0.222 | 0.710 ± 0.146 |
> | Δ ResNet-152V2 | −0.067 [−0.078, −0.056] | −0.052 [−0.067, −0.038] |

### [7.2] Section 4.3: ย่อหน้า manipulation check (หน้า 15 บรรทัดที่ประมาณ 1–3) เชื่อมกับ 4.4

**เดิม**
> This accuracy is a manipulation check: it verifies that each backbone acquired flip-sensitivity, but does not quantify how much downstream-relevant flip-bias was removed, and accordingly does not predict the magnitude of downstream gain.

**ใหม่**
> This accuracy is a manipulation check: it verifies that each backbone acquired flip-sensitivity, but does not quantify how much downstream-relevant flip invariance was reduced, and accordingly does not predict the magnitude of downstream gain; Section 4.4 measures that reduction directly on the ultrasound images.

---

## [8] A5: Compute cost (กรรมการข้อ 18)

### ผลที่ได้ (วัดจาก wall-clock ใน TensorBoard log จริง)

| Stage | GPU | Epochs | เวลา/epoch | รวม |
|---|---|---|---|---|
| Unlearning R1 (mini-ImageNet, 42,000 pairs, FC) | RTX 3090 Ti | 200 | 3,373 s | 187.4 h |
| Unlearning R2 (unfreeze block4–7, checkpoint ที่ใช้ = epoch 50) | RTX 3090 Ti | 50 | 3,277 s | 45.5 h |
| **Unlearning รวม** | | | | **≈ 233 GPU-h** |
| Downstream R1 (USAI, FC) | RTX 2080 Ti | 200 | 242 s | 13.5 h |
| Downstream R2 (block5a–7) | RTX 2080 Ti | 200 | 258 s | 14.3 h |
| **Downstream รวม (ต่อ 1 run)** | | | | **≈ 28 GPU-h** |

### [8.1] Section 4.2: เพิ่มย่อหน้า compute cost

**ใหม่ (เพิ่มท้ายย่อหน้า Experiment Environments and Settings)**
> **Computational cost.** Wall-clock times recorded in the training logs show that the unlearning stage for EfficientNet-B5 took 187 h for the first step (200 epochs over 42,000 image pairs) and 46 h for the second step (the 50 epochs up to the checkpoint used downstream) on a single RTX 3090 Ti, about 233 GPU-hours in total. Downstream fine-tuning on the ultrasound data (200 + 200 epochs over 4,601 training images) took about 28 h on a single RTX 2080 Ti. The unlearning stage is therefore the dominant cost, but it is performed once per backbone and the resulting weights are reused for every downstream task and run.

> หมายเหตุ: เวลา unlearning ใช้เครื่อง 28 (3090 Ti) ส่วน downstream ใช้เครื่อง 29 (2080 Ti) จึงระบุ GPU กำกับทุกตัวเลข · ถ้าทำ X1-C1 แบบ compute-matched ให้อ้างตัวเลขนี้เป็นงบ compute

---

## ข้อที่ต้องตรวจซ้ำก่อนส่ง

- [ ] ให้เลข Table A / B / (C ของ A3) ใหม่ และแก้ทุกจุดที่อ้าง "Table 5 / 6 / 7" ที่ถูกเลื่อน
- [ ] ทำ Supplementary Table S[x] = per-class sensitivity / specificity (ไฟล์ `A2_per_class.csv`)
- [x] Section 4.4 [A3] เพิ่มแล้ว (ใน [7.1]) และอ้างจาก [5.6]
- [ ] grep หา em-dash (`---`, U+2014) ให้เป็น 0
