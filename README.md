# IIT Bombay Crackathon Hackathon
![694ac7fc1c0c1_crackathon](https://github.com/user-attachments/assets/981cf5e6-bc77-4e3c-b924-835ee94100ca)

# High-Resolution Road Damage Detection

---

### 1. Executive Summary

This report summarizes our pipeline for detecting road damage in the Crackathon dataset using a **YOLOv8-Medium** model trained at **1024px** to address the thin crack problem.

By engineering a **Continuous Relay** training strategy to overcome 11GB GPU memory limits and employing **Test-Time Augmentation (TTA)** with resolution upscaling, the model achieved **0.573 mAP@50**. The final system demonstrates exceptional robustness, specifically boosting Pothole detection by **+8.5%** over the baseline.

### 2. Exploratory Data Analysis (EDA) & Data Strategy

We performed EDA to identify specific challenges in detecting small, irregular road defects and implemented evidence-driven solutions.

### 2.1 Visual Inspection and Data Sanity

Random sampling of the training data confirmed that defects like `Longitudinal Crack` (ID 0) and `Transverse Crack` (ID 1) are often only 2–3 pixels wide. Standard 640px resolution causes these features to collapse into noise.

<img src="IIT_BOMBAY/combined_cracks.jpg" width="800 percent" height="250 percent">

<p style="text-align: center;">
<em><strong>Figure 1</strong>: Sample image showing all <strong>five road damage classes</strong> with ground-truth boxes.</em>
</p>

### 2.2 Class Distribution & "The Pothole Problem"

A statistical audit revealed a severe class imbalance. While cracks were common, Pothole (ID 4) represented less than **10%** of the dataset instances.

* **Design Choice:** We implemented **Copy-Paste Augmentation (prob=0.3)**.
* **Justification:** We physically segmented potholes from training images and pasted them onto different road backgrounds. This artificially increased the effective sample size of the rarest class by 30%, directly leading to a recall boost.

### 2.3 The "Small Object" Constraint

Standard YOLO models resize images to 640px. Our analysis showed that at this resolution, longitudinal cracks lose connectivity, appearing as scattered noise rather than a continuous defect.

* **Design Choice:** We forced a fixed input resolution of **1024px** during training.
* **Justification:** High resolution is non-negotiable for this domain. To handle the increased VRAM usage on T4 GPUs (16GB), we utilized **Gradient Accumulation (nbs=16)** to simulate a batch size of 16 while physically fitting only 8 images per batch.

## 3. Methodology & System Architecture

### 3.1 Architecture Selection: YOLOv8-Medium

We selected **YOLOv8-Medium** over Small or Large variants.

* **Vs. Small:** Cracks require deep feature extraction to distinguish from shadows and pavement seams. Small models lacked the parameter depth for this texture differentiation.
* **Vs. Large:** The Large model required excessive VRAM, forcing a batch size too small for stable Batch Normalization. Medium offered the optimal trade-off for our constrained relay infrastructure.

### 3.2 The "Continuous Relay" Training Strategy

Due to Google Colab time limits, we could not train continuously. We engineered a **Continuous Relay** system:

1. **Legs 1–5 (Relay):** Short 6-epoch sprints with frequent resets to learn basic features quickly.
2. **Legs 6–11 (Continuous):** A pause-and-resume logic preserved optimizer state, simulating a single long run and allowing cosine annealing to converge properly.

## 4. Experimental Results & Analysis

We compared the initial relay approach against the final continuous pipeline.

### 4.1 Series A: The Relay Phase (Legs 1–5)

* **Run ID:** `Run_HighRes_Leg1` to `Run_HighRes_Leg5`
* **Strategy:** High learning rate (0.01) with aggressive Mosaic augmentation.
* **Outcome:** The model learned quickly, reaching **0.539 mAP**. However, frequent momentum resets prevented fine-grained convergence.

<table class="no-border">
<tr>
<td><img src="IIT_BOMBAY/assets/Final_PR_Curve.png" width="545 percent" height="170 percent"></td>
<td><img src="IIT_BOMBAY/Run_Continuous_Final\Final_Result\Normalized_Confusion_Matrix.png" width="545 percent" height="170 percent"></td>
</tr>
</table>

***Figure 2-3**: **PR curves** (left) and **normalized confusion matrix** (right) summarizing per-class performance and overall detection behavior.*


### 4.2 Series B: The Continuous Champion (Legs 6–11)

* **Run ID:** `Run_Continuous_Final`
* **Strategy:** Mosaic augmentation disabled at Epoch 24 to refine performance on realistic images.
* **Result:** The model achieved **0.573 mAP@50** and **0.303 mAP@50–95**, delivering its strongest overall performance.

<img src="IIT_BOMBAY/assets/results.png" width="430 percent" height="210 percent">

***Figure 4**: Full **training history** (Epochs 1–36) showing stable convergence after the relay-to-continuous transition.*

### 4.3 Summary of Metrics
<table class="metrics-table">
<tr>
  <th>Run ID</th>
  <th>Strategy</th>
  <th>mAP@50</th>
  <th>mAP@50–95</th>
  <th>Key Insight</th>
</tr>
<tr>
  <td>Run_HighRes_Leg5</td>
  <td>Relay (Reset LR)</td>
  <td>0.539</td>
  <td>0.285</td>
  <td>Baseline. Fast initial learning but plateaued.</td>
</tr>
<tr>
  <td>Run_Continuous_Leg9</td>
  <td>Continuous (Mosaic ON)</td>
  <td>0.548</td>
  <td>0.288</td>
  <td>Stability. Continuous training improved convergence.</td>
</tr>
<tr>
  <td>Run_Continuous_Leg11</td>
  <td>Mosaic Disabled</td>
  <td>0.571</td>
  <td>0.297</td>
  <td>Precision. Mosaic OFF enabled fine-tuning on real images.</td>
</tr>
<tr>
  <td><strong>Run_Continuous_Final</strong></td>
  <td><strong>Inference @ 1280px + TTA</strong></td>
  <td><strong>0.573</strong></td>
  <td><strong>0.303</strong></td>
  <td><strong>Deployment. Digital Zoom and TTA maximized final score.</strong></td>
</tr>
</table>

<div style="text-align: center;">
  <em><strong>Table 1</strong>: Progressive performance gains across training stages, highlighting the impact of <strong>continuous training</strong> and <strong>inference-time optimizations</strong>.</em>
</div>


## 5. Inference & Deployment Pipeline

To maximize leaderboard performance without retraining, we engineered a robust inference pipeline.

### 5.1 Resolution Upscaling ("Digital Zoom")

While training used 1024px, inference was performed at **1280px**, acting as a magnification step that stabilizes recall on the thinnest cracks.

### 5.2 Test-Time Augmentation (TTA)

We enabled TTA (`augment=True`), performing inference on flipped and scaled variants of each image and averaging predictions to reduce noise.

<table class="no-border">
<tr>
<td><img src="IIT_BOMBAY/assets/Set2_Batch_ACTUAL.jpg" width="545 percent" height="100 percent"></td>
<td><img src="IIT_BOMBAY/assets/Set2_Batch_PREDICTED.jpg" width="545 percent" height="100 percent"></td>
</tr>
</table>

***Figure 5–6**: Ground truth (left) versus model predictions (right), demonstrating robust detection of complex transverse cracks.*

<br>

## 6. Conclusion

We developed a high-resolution detection system capable of identifying microscopic road damage.

1. **Resolution is critical:** Scaling to 1024px and 1280px was the primary driver of performance.
2. **Imbalance corrected:** Copy-paste augmentation raised pothole detection by **+8.5%**.
3. **Infrastructure engineering:** The Continuous Relay approach enabled long-horizon training on free-tier hardware without numerical compromise.

<br>

Our final submission, `submission.zip`, generated using the `Run_Continuous_Final` model with TTA enabled, is fully compliant with the competition format.

<br>

## 7. Project Resources and Drive Repository

- **Dataset:** [Crackathon Dataset Link](https://www.kaggle.com/datasets/2bbd360e1ca39095fb6c6be9a26acbdc759db6ea5b5d75406451230e8ab42260)
- **Google Drive Repository:** [Google Drive Link](https://drive.google.com/drive/folders/13U0GS5NWBTu9TaoZhRu7DyNA6qhhRGXl?usp=sharing)
- **Models & Weights:** [Crackathon/01_models/Run_Continuous_Final](https://drive.google.com/drive/folders/1KH-cmVoj1V_pwM34wYmNPlC-iLjPfGlM?usp=drive_link)
- **Report Assets:** [Crackathon/03_reports/assets](https://drive.google.com/drive/folders/1sRDFKXKIvBpujFTpe5Y-ptJ7QUcoOrMw?usp=drive_link)
- **Source Code:** [Crackathon/00_notebooks](https://drive.google.com/drive/folders/1WaMS3LAW_RUmFK7LBkoObM0lCsBphOTh?usp=drive_link)
- **Submission File:** [Crackathon/02_submissions/submission.zip](https://drive.google.com/drive/folders/19qMFjPXhIh4xlVEhgKXtIMLIBkiFW4JZ?usp=drive_link)

---

<div align="center">

## 👥 Team

**Team Name**: ASHSUM
<br>
**Team Members**:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/ashir1s.png" width="150px;" alt="Ashirwad Sinha"/><br/>
      <a href="https://github.com/ashir1s">Ashiwad Sinha</a>
    </td>
    <td align="center">
      <img src="https://github.com/5umitpandey.png" width="150px;" alt="Sumit Pandey"/><br/>
      <a href="https://github.com/5umitpandey">Sumit Pandey</a>
    </td>
  </tr>
</table>

</div>

