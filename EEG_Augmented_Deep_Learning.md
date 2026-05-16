## Page 1

&lt;img&gt;electronics&lt;/img&gt;
&lt;img&gt;MDPI&lt;/img&gt;

Article
# High-Resolution Time-Frequency Feature Selection and EEG Augmented Deep Learning for Motor Imagery Recognition

Mouna Bouchane <sup>1</sup>&lt;img&gt;ID&lt;/img&gt;, Wei Guo <sup>1,*</sup>&lt;img&gt;ID&lt;/img&gt; and Shuojin Yang <sup>2</sup>&lt;img&gt;ID&lt;/img&gt;

1 Key Laboratory of Augmented Reality, School of Mathematical Sciences, Hebei Normal University, Shijiazhuang 050024, China; mouna.bouchane@gmail.com
2 Department of Computer Science and Technology, Tsinghua University, Beijing 100084, China; yangshuojin@tsinghua.edu.cn
* Correspondence: guowei@chmiot.net

## Abstract
Motor Imagery (MI) based Brain Computer Interfaces (BCIs) have promising applications in neurorehabilitation for individuals who have lost mobility and control over parts of their body due to brain injuries, such as stroke patients. Accurately classifying MI tasks is essential for effective BCI performance, but this task remains challenging due to the complex and non-stationary nature of EEG signals. This study aims to improve the classification of left and right-hand MI tasks by utilizing high-resolution time-frequency features extracted from EEG signals, enhanced with deep learning-based data augmentation techniques. We propose a novel deep learning framework named the Generalized Wavelet Transform-based Deep Convolutional Network (GDC-Net), which integrates multiple components. First, EEG signals recorded from the C3, C4, and Cz channels are transformed into detailed time-frequency representations using the Generalized Morse Wavelet Transform (GMWT). The selected features are then expanded using a Deep Convolutional Generative Adversarial Network (DCGAN) to generate additional synthetic data and address data scarcity. Finally, the augmented feature maps data are subsequently fed into a hybrid CNN-LSTM architecture, enabling both spatial and temporal feature learning for improved classification. The proposed approach is evaluated on the BCI Competition IV dataset 2b. Experimental results showed that the mean classification accuracy and Kappa value are 89.24% and 0.784, respectively, making them the highest compared to the state-of-the-art algorithms. The integration of GMWT and DCGAN significantly enhances feature quality and model generalization, thereby improving classification performance. These findings demonstrate that GDC-Net delivers superior MI classification performance by effectively capturing high-resolution time-frequency dynamics and enhancing data diversity. This approach holds strong potential for advancing MI-based BCI applications, especially in assistive and rehabilitation technologies.

## Keywords: motor imagery; brain-computer interface; electrocephalography; wavelet transform; generative adversarial network; convolutional neural network

&lt;img&gt;check for updates&lt;/img&gt;
Received: 30 April 2025
Revised: 7 July 2025
Accepted: 8 July 2025
Published: 14 July 2025

**Citation:** Bouchane, M.; Guo, W.; Yang, S. High-Resolution Time-Frequency Feature Selection and EEG Augmented Deep Learning for Motor Imagery Recognition. *Electronics* **2025**, *14*, 2827. https://doi.org/10.3390/electronics14142827

**Copyright:** © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).

## 1. Introduction
A Brain Computer Interface (BCI) enables direct communication between users and external devices by decoding non-invasive Electroencephalography (EEG) signals [1]. Among these, motor imagery (MI) is the mental simulation of movement without actual muscular contraction that produces distinctive event-related desynchronization (ERD) and synchronization (ERS) patterns that mirror real motor execution [2]. Traditional approaches to

Electronics 2025, 14, 2827
https://doi.org/10.3390/electronics14142827

---


## Page 2

<header>Electronics 2025, 14, 2827</header>
&lt;page_number&gt;2 of 14&lt;/page_number&gt;

MI-EEG decoding typically involve a multi-step process that begins with rigorous preprocessing to remove noise and filter out irrelevant frequency bands. Once cleaned, the signal is analyzed using various mapping techniques such as Common Spatial Pattern (CSP) [3], Filter Bank Common Spatial Pattern (FBCSP) [4], Short-Time Fourier Transform (STFT) [5], and Wavelet Transform (WT) [6], to extract discriminative features in the frequency domain. These features are then classified using machine learning algorithms. However, this sequential pipeline is prone to error propagation; inaccuracies at any intermediate stage can adversely affect the final decoding performance. Additionally, the inherently weak EEG signals are often obscured by uncorrelated biological noise, and variability in physiological structures across subjects and sessions introduces significant differences in feature distribution. These challenges underscore the need for optimized signal extraction and robust feature modeling to enhance the practical applicability of BCI systems.

In recent years, deep neural networks (DNNs) have become increasingly popular for MI-EEG recognition. Their end-to-end architecture and automatic feature extraction capabilities help reduce redundant information and enhance classification performance. Nevertheless, practical challenges persist due to the difficulty of acquiring large datasets limited by subject availability, experimental duration, and operational complexity. Since DNNs are highly sensitive to the volume of training data, small datasets often result in poor generalization, ultimately undermining classification accuracy [7]. In the context of MI applications within BCIs, traditional data augmentation (DA) techniques have been employed to enhance the performance of classifiers by addressing issues such as class imbalance and limited training data. Typical methods of DA include geometric transformation (GT), noise addition (NA) [8], and generative models [9,10]. The geometric transformation strategy augments the training dataset by applying geometric transformations, such as reflection, rotation, shear, and shift, to the training images, thereby increasing the number of available samples. However, these transformations may produce unrealistic samples that do not accurately represent natural variations in the data, potentially misleading the model. Overly aggressive transformations can introduce artifacts that compromise the model’s generalization ability. Finally, determining the optimal transformation parameters is challenging, as insufficient modification may yield limited benefits while excessive changes may degrade performance. As another typical DA strategy, NA includes additive Gaussian noise, uniform noise, and signal-specific noise tailored to mimic real-world artifacts. These methods enhance training data diversity and improve model robustness by exposing classifiers to a broader range of signal variability.

Nonetheless, excessive noise may obscure essential MI features, leading to reduced classification accuracy [11]. Moreover, careful parameter tuning is required to balance augmentation benefits with the risk of signal distortion. Recently, studies demonstrated that the Generative Adversarial Networks (GANs) in MI-BCI applications succeed in generating artificial EEG data, which can enhance the training process and improve classification outcomes [12]. GANs address the challenges of data scarcity and enhance the quality of generated data while maintaining spatial features specifically for cross-subject classification. Despite these advancements, the exploration of MI signal representation and generation remains limited [13].

This work introduces the generalized Wavelet Transform-based Deep Convolutional Network (GDC-Net), a novel deep learning framework for MI-EEG classification. Designed to address key limitations of conventional pipelines, GDC-Net combines signal transformation, data augmentation, and spatiotemporal modeling in a unified architecture. The framework was evaluated using the publicly available BCI Competition IV 2b dataset, where it outperformed existing state-of-the-art approaches.

The main contributions of this work are as follows:

---


## Page 3

Electronics 2025, 14, 2827
&lt;page_number&gt;3 of 14&lt;/page_number&gt;

(1) Enhanced time-frequency representation: GDC-Net utilizes GMWT to generate high-resolution time-frequency scalograms, capturing MI-related EEG features more effectively than traditional wavelet-based techniques.
(2) Data augmentation: To mitigate the problem of data scarcity, the framework incorporates DCGAN, which produces synthetic scalogram images, increasing the diversity and volume of the training data.
(3) Optimized CNN-LSTM architecture: For classification, GDC-Net employs a hybrid model that combines a 2D CNN for spatial feature extraction and LSTM units for learning temporal dynamics within the EEG signals.

GDC-Net achieved a classification accuracy of 89.24%, surpassing previous works on the BCI Competition IV 2b dataset and establishing a new benchmark for MI-EEG decoding in BCI. This study presents a novel and effective approach for MI-EEG classification, contributing to the advancement of robust BCI systems for real-world applications.

The structure of this paper is as follows: Section 2 details the architecture and methodology behind GDC-Net, including an in-depth discussion of features selection, extraction, and classification. Section 3 evaluates the model’s performance in distinguishing between left- and right-hand motor tasks. Lastly, Sections 4 and 5 discuss the results compared with state-of-the-art methods and provide concluding insights.

## 2. Materials and Methods

### 2.1. Dataset Description

We employed the BCI Competition IV 2b Dataset [14] to assess our proposed GDC-Net for MI classification tasks. EEG recordings were acquired from nine subjects via three electrodes placed at positions C3, Cz, and C4, sampled at 250 Hz. Each participant underwent five sessions: the first two contained 120 trials each, while the last three comprised 160 trials per session, yielding 720 trials per subject. Altogether, the dataset included 6480 trials across all subjects. Some trials within all sessions exhibited contamination from eye blinks. The motor imagery tasks involved in this dataset included imagined movements of both the left and right hands. Figure 1 describes each trial’s experimental protocol and timing sequence.

&lt;img&gt;Figure 1. The experimental procedure consisted of sequential trials where participants controlled a gray face icon on the screen using MI. At the beginning of each trial, a directional cue prompted the subject to imagine either a left or right-hand movement to steer the face accordingly. If the imagined movement correctly matched the cue, a green smiley face appeared as positive feedback; otherwise, a red sad face indicated an incorrect response. At 7.5 s, the cue disappeared, and the screen turned blank. Before the subsequent trial, a random inter-trial interval of 1 to 2 s was introduced for preparation and reset.&lt;/img&gt;
| Phase | Start Time (s) | End Time (s) | Description |
| :--- | :--- | :--- | :--- |
| Smiley (grey) | 0 | 3 | Preparation with grey face icon |
| Beep | 2 | — | Acoustic signal |
| Cue | 3 | 4 | Directional cue (left or right) |
| Imaging with feedback | 4 | 7.5 | Motor imagery with green/red face feedback |
| Pause | 7.5 | 8 | Screen turns blank |
| Inter-trial interval | 8 | 9–10 | Random interval (1–2 s) for reset |

### 2.2. Design of GDC-Net Processing Pipeline

GDC-Net is an end-to-end deep learning framework structured around three core components for MI-EEG classification: feature extraction using Wavelet Transforms, data augmentation through generative modeling, and a deep learning-based classification module. The overall architecture is depicted in Figure 2. The MI-EEG signal processing pipeline begins with band-pass filtering to focus on the low-frequency components most

---


## Page 4

Electronics 2025, 14, 2827
&lt;page_number&gt;4 of 14&lt;/page_number&gt;

relevant to MI tasks. As established in the literature, the mu rhythm (8–12 Hz) and beta rhythm (18–26 Hz) contain essential MI-related features. We used beta band extension (13–30 Hz) to ensure the retention of relevant neural activity for classification [15]. The selected time interval [3.5 s, 7.5 s] corresponds to the period where subjects actively imagine movement direction (left or right). Following preprocessing, GMWT is applied to the C3, Cz, and C4 electrodes, generating high-resolution time-frequency scalograms that effectively capture transient MI patterns. DCGAN generates synthetic time-frequency representations to address limited data training issues, enhancing data diversity and improving model generalization. The enriched data is then processed through a hybrid deep learning model. In this configuration, CNN is responsible for extracting discriminative features from the generated scalograms, which are then forwarded to the LSTM network to capture temporal dependencies and classify the motor imagery task as either left- or right-hand movement. CNN is selected for its ability to automatically learn multi-scale spatial features from input data, eliminating the need for manual feature engineering. LSTM, on the other hand, is integrated for its effectiveness in preserving long-range temporal information throughout the sequence, making it well-suited for the final classification step.

&lt;img&gt;Figure 2. Flowchart of the proposed GDC-Net framework for MI-EEG signal processing. The dashed box highlights the core algorithm, which integrates high-resolution feature extraction using GMW Transform, data augmentation based on DCGAN, and spatio-temporal modeling using a hybrid CNN-LSTM architecture.&lt;/img&gt;
| Block Name | Function / Description | Connection To |
| :--- | :--- | :--- |
| Raw EEG | Input signal source | Filter (8–30Hz) |
| Filter (8–30Hz) | Band-pass filtering of EEG signal | GMW Transform |
| GMW Transform | Time-Frequency Scalograms generation | DCGAN Augmentation |
| DCGAN Augmentation | Synthetic Scalograms generation | CNN (Feature Extraction) |
| CNN | Feature Extraction from scalograms | LSTM (Temporal Modeling) |
| LSTM | Temporal Modeling of features | Left Hand / Right Hand |
| Left Hand | Classification output: Left-hand movement | — |
| Right Hand | Classification output: Right-hand movement | — |

2.3. TimeFrequency Representation Using Generalized Morse Wavelet Transform

The GDC-Net feature selection process begins with the application of the Continuous Wavelet Transform (CWT), which serves as an effective tool for analyzing MI-EEG data. Given the non-stationary and transient characteristics of EEG signals, CWT offers a detailed time-frequency representation, enabling the identification of both spectral and temporal variations. This capability is particularly advantageous for capturing subtle fluctuations in EEG signals, which conventional techniques might fail to detect. CWT enhances feature extraction by retaining key signal dynamics, leading to improved EEG analysis and a more refined understanding of brain activity.

In this work, time-frequency analysis is performed using CWT, where GMWT is employed as the mother wavelet. This selection provides adaptive control over time-frequency localization for motor imagery EEG feature extraction [16]. GMWT represents a class of analytic wavelets that support only the positive real axis for Fourier transformations of complex-valued wavelets. These analytic wavelets are particularly effective for analyzing modulated signals characterized by time-varying amplitude and frequency. While non-analytic wavelets enhance sharp transitions in the time-frequency domain, analytic wavelets are more suited for capturing the oscillatory behavior of frequency transients, pro-

---


## Page 5

Electronics 2025, 14, 2827
&lt;page_number&gt;5 of 14&lt;/page_number&gt;

providing a comprehensive representation of EEG signal dynamics [17]. Moreover, GMWT’s parametric flexibility enables fine-tuning of symmetry and concentration properties, ensuring alignment with the morphological features of MI signals.

The analytic GMWT is characterized in the frequency domain by its Fourier Transform:
$$
\hat{\psi}(\omega) = U(\omega)a_{\beta,\gamma}\omega^{\beta}e^{-\omega\gamma} \quad (1)
$$
In this expression, $U(\omega)$ denotes the unit step function, ensuring the wavelet is defined only for positive frequencies; $a_{\beta,\gamma}$ is the normalization constant that guarantees the wavelet has unit energy; $\beta$ and $\gamma$ are parameters that shape the wavelet’s time-frequency characteristics; and $\omega$ represents the angular frequency. The time-domain representation $\psi(t)$ is derived by applying the inverse Fourier Transform to $\hat{\psi}(\omega)$. This formulation results in an analytic wavelet with adjustable time-frequency localization properties, making it particularly effective for analyzing non-stationary signals such as EEG data [18]. The GMWT parameters $\beta$ and $\gamma$ are crucial in adjusting the wavelet’s time-frequency characteristics to match the specific features of EEG data. The parameter $\beta$ affects the wavelet’s time-bandwidth product, influencing the balance between time and frequency resolution. Higher $\beta$ values improve frequency localization but may reduce temporal precision. On the other hand, $\gamma$ controls the symmetry of the wavelet in time. Choosing appropriate $\beta$ and $\gamma$ values is essential for effectively capturing the transient and oscillatory components of motor imagery-related EEG signals, thereby enhancing the performance of brain-computer interface systems [16]. In this paper, $\gamma$ and bandwidth were set at 3 and 60, respectively, to prevent feature loss and avoid excessive spectral smoothing.

### 2.4. Synthetic Scalogram Generation Using DCGAN

As part of the GDC-Net framework, DCGAN is employed to augment the training dataset by generating synthetic time-frequency representations, thereby addressing the challenge of limited EEG data and enhancing model generalization.

DCGAN consists of a generator $G$, which synthesizes EEG scalograms from a latent distribution, and a discriminator $D$, which classifies inputs as real or synthetic [19]. The generator learns to synthesize EEG scalograms by mapping a random latent vector $z \sim N(0, I)$ from a multivariate normal distribution to a high-dimensional feature space:
$$
\hat{x} = G(z; \theta_G), \quad (2)
$$
where $\theta_G$ represents the trainable parameters of the generator. The discriminator $D$, parameterized by $\theta_D$, acts as a classifier that maps an input EEG scalogram $x$ to a probability score:
$$
D(x) : x \to [0, 1], \quad (3)
$$
which determines whether the input is real or synthetic. The discriminator is optimized to maximize its ability to correctly classify real ($x \sim p_{data}(x)$) and generated samples ($G(z)$) such that:
$$
D(G(z)) = p(real). \quad (4)
$$
The adversarial training follows a minimax optimization process where the generator and discriminator are trained to compete against one another. The objective function is formulated as:
$$
Min_G Max_D \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \quad (5)
$$

---


## Page 6

Electronics 2025, 14, 2827
&lt;page_number&gt;6 of 14&lt;/page_number&gt;

This equation represents the adversarial training process, where the discriminator D maximizes its ability to correctly classify real (x) and generated (G(z)) samples, while the generator G minimizes this loss, attempting to generate samples that D classify as real.

The discriminator is optimized by minimizing its loss function, ensuring it effectively distinguishes real EEG samples from synthetic ones:
$$L_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))] \quad (6)$$

Simultaneously, the generator is updated by minimizing:
$$L_G = -\mathbb{E}[\log D(z)] \quad (7)$$

which forces it to generate increasingly realistic EEG scalograms capable of fooling the discriminator. These updates occur iteratively using stochastic gradient descent (SGD) with learning rate α, through backpropagation:
$$\theta_D \leftarrow \theta_D - \alpha \nabla_{\theta_D} L_D, \theta_G \leftarrow \theta_G - \alpha \nabla_{\theta_G} L_G \quad (8)$$

The adversarial training process progresses iteratively until the generator is capable of producing synthetic EEG representations that closely resemble real data, thereby enhancing the model’s robustness and improving classification performance for the MI task recognition. In this work, the baseline DCGAN architecture was adapted from the framework proposed by [20]. Both the generator and discriminator components were further refined to better extract relevant features specific to the MI-EEG dataset.

### 2.5. CNN-LSTM Spatiotemporal Modeling for Motor Imagery Classification

The classification component of the GDC-Net pipeline integrates a 2D CNN and an LSTM network to effectively extract and model spatiotemporal features from EEG signals. The CNN is responsible for capturing EEG channel connectivity patterns, generating feature representations that are subsequently processed by the LSTM network to learn temporal dependencies in the data. The hybrid architecture enables GDC-Net to classify MI tasks into “Left” and “Right” categories with improved accuracy. LSTM, a specialized variant of RNNs, introduces memory cells that mitigate the vanishing and exploding gradient problems, allowing the network to retain relevant information over extended time sequences. Each LSTM unit consists of three gating mechanisms: the forget gate, input gate, and output gate, which regulate information flow and update the cell state dynamically. The forget gate determines the retention or removal of past information:
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad (9)$$

where $f_t$ is the forget gate activation, $\sigma$ is the sigmoid activation function, and $W_f$, $U_f$ represent weight matrices applied to the current input $x_t$ and the previous hidden state $h_{t-1}$. The input gate controls the addition of new information:
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad (10)$$

with the candidate cell state computed as:
$$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad (11)$$

---


## Page 7

Electronics 2025, 14, 2827
&lt;page_number&gt;7 of 14&lt;/page_number&gt;

The cell state update is then formulated as:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad (12)$$
where $\odot$ denotes element-wise multiplication. Finally, the output gate determines the extent to which the updated cell state contributes to the final hidden state:
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad (13)$$
$$h_t = o_t \odot \tanh(C_t) \quad (14)$$

Through this gated structure, LSTM effectively captures long-range dependencies in EEG signals, preserving crucial patterns associated with motor imagery tasks. The integration of CNN and LSTM ensures that the model leverages both spatial feature extraction and temporal sequence learning, improving classification accuracy. The CNN-LSTM architecture illustrated in Figure 3 is trained using cross-validation, where each fold consists of 90% training data and 10% validation data to ensure generalization. The hyperparameters, optimization techniques, and regularization strategies are detailed in Table 1.

&lt;img&gt;Figure 3. Architecture of the hybrid CNN-LSTM classifier in GDC-Net for MI-EEG recognition.&lt;/img&gt;
| Block Name | Details / Components |
| :--- | :--- |
| EEG Signal (Subject 1) | Raw input signals |
| CWT Data | 200 x 93 x 1 |
| 2D Convolutional layer | Conv 1, Max Pooling, Conv 2, Max Pooling, Conv 3, Max Pooling |
| LSTM layer | Two-step temporal sequence processing (h<sub>t-1</sub>, h<sub>t</sub>) |
| Fully Connected | Dense layer connecting LSTM output to classification |
| Softmax layer | Output classes: Right, Left |

Table 1. Hyperparameter settings and training details of the GDC-Net classification module.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Hyperparameter</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">CNN architecture</td>
      <td>No. of convolutional layers</td>
      <td>3 Conv2D layers</td>
    </tr>
    <tr>
      <td>Kernel Size</td>
      <td>3 x 3</td>
    </tr>
    <tr>
      <td>Activation function</td>
      <td>RELU</td>
    </tr>
    <tr>
      <td>Pooling</td>
      <td>Max-Pooling (2 x 2)</td>
    </tr>
    <tr>
      <td rowspan="4">LSTM Temporal modeling</td>
      <td>No. of LSTM Layers</td>
      <td>2 LSTM Layers</td>
    </tr>
    <tr>
      <td>Hidden layers</td>
      <td>128 units for the 1st LSTM<br>64 units for the 2nd LSTM</td>
    </tr>
    <tr>
      <td>Activation function</td>
      <td>tanh</td>
    </tr>
    <tr>
      <td>Dropout rate</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td rowspan="2">Fully connected layer</td>
      <td>No of dense layers</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Final Classification Layer</td>
      <td>Softmax (2 Classes: Left/Right)</td>
    </tr>
    <tr>
      <td rowspan="4">Optimization & Training</td>
      <td>Optimizer</td>
      <td>Adam</td>
    </tr>
    <tr>
      <td>L2 regularization, Weight Decay</td>
      <td>0.0001</td>
    </tr>
    <tr>
      <td>Mini-Batch algorithm</td>
      <td>Gradient decent</td>
    </tr>
    <tr>
      <td>Cross-validation strategy</td>
      <td>10-Fold cross-validation</td>
    </tr>
  </tbody>
</table>

---


## Page 8

<header>Electronics 2025, 14, 2827</header>
&lt;page_number&gt;8 of 14&lt;/page_number&gt;

The proposed CNN-LSTM model is designed to efficiently classify MI tasks using time-frequency representations extracted from EEG signals. Initially, the EEG signals recorded from electrodes C3, Cz, and C4 undergo CWT using the GMW transform to obtain high-resolution time-frequency scalograms. To ensure uniform feature representation, the extracted mu (8–13 Hz) and beta (14–30 Hz) bands are resized using cubic spline interpolation, resulting in standardized input 2D time-frequency images of (93 × 200), where 93 represents stacked frequency components across the three electrodes, and 200 corresponds to resampled time points (Figure 3).

The CNN module employs 2D convolutional layers to extract spatial features from the data augmented images. It consists of three convolutional layers with increasing filter sizes (32, 64, and 128), each followed by batch normalization to stabilize training and max pooling (2 × 2) to progressively reduce dimensionality. The resulting feature maps are flattened and reshaped into sequential form (25 × 1408) before being passed to the LSTM module, which captures temporal dependencies. The LSTM network consists of two layers (128 and 64 units, respectively), which effectively model the sequential nature of EEG signals. To enhance generalization, a fully connected layer (64 neurons) with dropout (0.5) is applied before classification. The final output is obtained using a softmax activation function, which assigns probabilities to the left- and right-hand MI classes. The model is trained using cross-entropy loss and optimized with the Adam optimizer for 150 epochs. The average training time is 485 s. The hybrid CNN-LSTM architecture efficiently integrates spatial and temporal information, ensuring robust MI classification while maintaining computational efficiency.

## 3. Performance Assessment and Results

The proposed GDC-Net algorithm was evaluated using the BCI Competition IV 2b Dataset. Experiments were conducted in Matlab 2022a and Python 3.8, leveraging TensorFlow 3.4.0 for model training and evaluation. The computational environment included an Intel® Core™ i5-7500 CPU (3.40 GHz) with 16 GB RAM (Intel Corporation, Santa Clara, CA, USA) and an NVIDIA GeForce GTX 1050 Ti GPU (Nvidia Corporation, Santa Clara, CA, USA), running on Windows 10 (64-bit). This setup ensured efficient execution of data preprocessing, feature extraction, and deep learning model training. Adaptive moment Estimation (Adam) is used to train the weights of the proposed CNN-LSTM algorithm.

To evaluate the impact of DCGAN-based data augmentation, we compared the classification performance of the proposed GDC-Net with and without augmentation across all subjects, as shown in Table 2. The evaluation considers Kappa values and classification accuracy, along with statistical measures, including mean and standard deviation. The calculation expression of the Kappa coefficient is as follows:

$$
Kappa = \frac{acc - p_e}{1 - p_e} \quad (15)
$$

where the accuracy $acc$ is defined as:

$$
acc = \frac{TP + TN}{TP + TN + FP + FN} \quad (16)
$$

where $TP$, $TN$, $FP$, and $FN$ represent true positives, true negatives, false positives, and false negatives, respectively. Since two-class classification is studied here, the random classification accuracy in Equation (15) is $p_e = 0.5$.

With DCGAN augmentation, the Kappa values ranged from 0.4938 (S3) to 0.9712 (S4), indicating substantial variation in model agreement across subjects. Similarly, accuracy

---


## Page 9

Electronics 2025, 14, 2827
&lt;page_number&gt;9 of 14&lt;/page_number&gt;

values spanned from 74.69% (S3) to 98.56% (S4). The mean Kappa value across all subjects was 0.7847, with a mean accuracy of 89.24%. The low standard deviations (0.1303 for Kappa, 8.09 for accuracy) suggest stable performance across subjects.

Without DCGAN augmentation, classification performance declined consistently across all subjects. The Kappa values dropped, ranging from 0.3940 (S3) to 0.9214 (S4), while accuracy decreased, with the lowest at 69.36% (S3) and the highest at 93.45% (S4). The mean Kappa and accuracy values decreased to 0.7187 and 84.32%, respectively, confirming the positive impact of DCGAN on model generalization.

The most substantial relative gains were observed for low-performing subjects, particularly S2 and S3, where Kappa increased by 21.43% and 25.45%, and accuracy improved by 6.82% and 7.17%, respectively. These enhancements demonstrate DCGAN’s ability to improve class separability in challenging cases and validate the effectiveness of DCGAN-based augmentation in enhancing classification robustness and generalization, particularly in subjects where the model otherwise struggles to capture MI patterns effectively.

The classification performance of the proposed GDC-Net was evaluated against the state-of-the-art MI-EEG classification techniques, including Anchored STFT-Skip-Net with GNAA [21], CapsNet [22], STFT-VGG16 [23], CutCat [24], CNN-MLP [25], and CWT-SCNN [26], ADFCNN [27], EEG-Conformer [28], and CLT-Net [29]. As presented in Table 3, our model consistently outperformed competing approaches, achieving a mean classification accuracy of 89.24%, surpassing CWT-SCNN (83.2%), CNN-MLP (74.80%), CutCat (78.44%), Anchored STFT-Skip-Net (76.0%), CapsNet (78.44%), STFT-VGG16 (73.81%), EEG-Conformer (84.63%), CLT-Net (87.11%), and ADFCNN (87.81%).

Table 2. Comparison of classification metrics, Kappa, and accuracy results with and without DCGAN data augmentation.

<table>
  <thead>
    <tr>
      <th rowspan="2">Subject</th>
      <th colspan="2">Without DCGAN</th>
      <th colspan="2">With DCGAN</th>
    </tr>
    <tr>
      <th>Kappa</th>
      <th>Accuracy</th>
      <th>Kappa</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S1</td>
      <td>0.7456</td>
      <td>85.98</td>
      <td>0.7962</td>
      <td>89.81</td>
    </tr>
    <tr>
      <td>S2</td>
      <td>0.4668</td>
      <td>73.11</td>
      <td>0.5666</td>
      <td>78.33</td>
    </tr>
    <tr>
      <td>S3</td>
      <td>0.3940</td>
      <td>69.36</td>
      <td>0.4938</td>
      <td>74.69</td>
    </tr>
    <tr>
      <td>S4</td>
      <td>0.9214</td>
      <td>93.45</td>
      <td>0.9712</td>
      <td>98.56</td>
    </tr>
    <tr>
      <td>S5</td>
      <td>0.8323</td>
      <td>91.30</td>
      <td>0.9324</td>
      <td>96.62</td>
    </tr>
    <tr>
      <td>S6</td>
      <td>0.7652</td>
      <td>85.69</td>
      <td>0.8100</td>
      <td>90.50</td>
    </tr>
    <tr>
      <td>S7</td>
      <td>0.6825</td>
      <td>81.56</td>
      <td>0.7380</td>
      <td>86.90</td>
    </tr>
    <tr>
      <td>S8</td>
      <td>0.8467</td>
      <td>90.34</td>
      <td>0.8906</td>
      <td>94.53</td>
    </tr>
    <tr>
      <td>S9</td>
      <td>0.8142</td>
      <td>88.12</td>
      <td>0.8638</td>
      <td>93.19</td>
    </tr>
    <tr>
      <td>Mean</td>
      <td>0.7187</td>
      <td>84.32</td>
      <td>0.7847</td>
      <td>89.24</td>
    </tr>
    <tr>
      <td>Std.</td>
      <td>0.1675</td>
      <td>7.78</td>
      <td>0.1303</td>
      <td>8.09</td>
    </tr>
  </tbody>
</table>

Table 3. The classification results of different algorithms on BCI Competition IV 2b dataset.

<table>
  <thead>
    <tr>
      <th>Subjects</th>
      <th>STFT-SkipNet-GNAA [21]</th>
      <th>CapsNet [22]</th>
      <th>STFT-VGG16 [23]</th>
      <th>CutCat [24]</th>
      <th>CNN-MLP [25]</th>
      <th>CWT-SCNN [26]</th>
      <th>ADFCNN [27]</th>
      <th>EEG-Conformer [28]</th>
      <th>CLT-Net [29]</th>
      <th>Proposed GDC-Net</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S1</td>
      <td>79.90</td>
      <td>78.75</td>
      <td>72.60</td>
      <td>75.31</td>
      <td>74.50</td>
      <td>74.70</td>
      <td>79.37</td>
      <td>73.13</td>
      <td>75.94</td>
      <td>89.81</td>
    </tr>
    <tr>
      <td>S2</td>
      <td>57.30</td>
      <td>55.71</td>
      <td>60.30</td>
      <td>60.00</td>
      <td>64.30</td>
      <td>81.30</td>
      <td>72.50</td>
      <td>67.50</td>
      <td>69.29</td>
      <td>78.33</td>
    </tr>
    <tr>
      <td>S3</td>
      <td>56.20</td>
      <td>55.00</td>
      <td>66.90</td>
      <td>60.31</td>
      <td>71.80</td>
      <td>68.10</td>
      <td>82.81</td>
      <td>79.06</td>
      <td>84.68</td>
      <td>74.69</td>
    </tr>
    <tr>
      <td>S4</td>
      <td>95.10</td>
      <td>95.93</td>
      <td>91.20</td>
      <td>97.19</td>
      <td>94.50</td>
      <td>96.30</td>
      <td>96.25</td>
      <td>97.19</td>
      <td>97.81</td>
      <td>98.56</td>
    </tr>
    <tr>
      <td>S5</td>
      <td>87.50</td>
      <td>83.12</td>
      <td>80.60</td>
      <td>82.81</td>
      <td>79.50</td>
      <td>92.50</td>
      <td>99.37</td>
      <td>96.88</td>
      <td>97.50</td>
      <td>96.62</td>
    </tr>
    <tr>
      <td>S6</td>
      <td>83.10</td>
      <td>83.43</td>
      <td>70.60</td>
      <td>82.50</td>
      <td>75.00</td>
      <td>86.90</td>
      <td>84.68</td>
      <td>83.13</td>
      <td>85.31</td>
      <td>90.50</td>
    </tr>
    <tr>
      <td>S7</td>
      <td>75.60</td>
      <td>75.62</td>
      <td>73.20</td>
      <td>74.69</td>
      <td>70.50</td>
      <td>73.40</td>
      <td>93.43</td>
      <td>93.13</td>
      <td>93.13</td>
      <td>86.90</td>
    </tr>
    <tr>
      <td>S8</td>
      <td>71.40</td>
      <td>91.25</td>
      <td>77.70</td>
      <td>88.13</td>
      <td>71.80</td>
      <td>91.60</td>
      <td>95.31</td>
      <td>92.81</td>
      <td>91.88</td>
      <td>94.53</td>
    </tr>
    <tr>
      <td>S9</td>
      <td>77.90</td>
      <td>87.18</td>
      <td>71.20</td>
      <td>85.00</td>
      <td>71.00</td>
      <td>84.40</td>
      <td>86.56</td>
      <td>90.00</td>
      <td>88.44</td>
      <td>93.19</td>
    </tr>
    <tr>
      <td>Mean</td>
      <td>76.0</td>
      <td>78.44</td>
      <td>73.81</td>
      <td>78.44</td>
      <td>74.80</td>
      <td>83.2</td>
      <td>87.81</td>
      <td>84.63</td>
      <td>87.11</td>
      <td>89.24</td>
    </tr>
  </tbody>
</table>

---


## Page 10

Electronics 2025, 14, 2827
&lt;page_number&gt;10 of 14&lt;/page_number&gt;

Our method demonstrated consistent improvements across subjects, particularly for S1 (89.81%), S5 (96.62%), S6 (90.50%), S7 (86.90%), S8 (94.53%), and S9 (93.19%), where it significantly outperformed competing approaches. The highest accuracy of 98.56% was recorded for S4, confirming the model’s ability to efficiently extract MI-related features. Furthermore, our method notably improved classification performance for lower-performing subjects, such as S2 (78.33%) and S3 (74.69%), achieving gains of 5–10% over the best alternative methods in these cases. Notable inter-subject performance variability was observed, with accuracies ranging from 74.69% (S3) to 98.56% (S4). This disparity reflects the well-documented phenomenon of inter-subject variability in MI-EEG classification, influenced by factors such as individual differences in cortical activation, motor imagery strategy, EEG signal quality, and cognitive engagement.

In contrast, CutCat and Anchored STFT-Skip-Net exhibited lower performance due to suboptimal time-frequency resolution and limited temporal modeling. Similarly, CapsNet and lacking data augmentation struggled with generalization, while CNN-MLP failed to capture spatial dependencies effectively. Although CWT-SCNN outperformed STFT-based approaches, it was still surpassed by the proposed GDC-Net model, which leverages adaptive wavelet-based feature extraction and deep spatiotemporal learning.

Recent studies have proposed increasingly sophisticated architectures for MI-EEG classification. The attention-based dual-scale fusion CNN (ADFCNN) reported a mean accuracy of 87.81% by extracting spectral and spatial features at multiple scales and integrating a self-attention mechanism to enhance feature fusion. Despite its strong performance, the model’s adaptability is limited by its reliance on fixed architectural parameters and the absence of data augmentation strategies to improve generalization.

The Conformer model combines convolutional layers with a Transformer module, effectively capturing local spatial patterns and global temporal dependencies. It achieved an accuracy of 84.63%, demonstrating the value of self-attention in EEG feature modeling. However, incorporating the attention mechanism led to a 17.6% increase in trainable parameters, primarily due to the linear transformations and feed-forward components used for capturing global dependencies, which reduces the model’s computational efficiency.

More recently, CLT-Net has advanced this hybrid design by integrating elements from both Conformer and EEGNet. It achieved a higher mean accuracy of 87.11% while maintaining a lower parameter count, thus improving efficiency. Nevertheless, the model exhibited high sensitivity to several hyperparameters, particularly token size, transformer depth, and the number of attention heads, which increases the need for extensive tuning. Furthermore, the segmentation and reconstruction (S&R) augmentation strategy employed in CLT-Net showed limited effectiveness in enhancing subject-specific decoding performance.

These results confirm that the integration of GMWT for adaptive time-frequency analysis, CNN for spatial feature encoding, and LSTM for temporal sequence modeling significantly enhances MI classification performance without requiring deep attention-based modules. Additionally, the use of DCGAN-based data augmentation ensures better generalization, mitigating dataset limitations without the need for manually engineered geometric perturbations.

Together, these components contribute to the robustness reinforcement of the proposed GDC-Net framework for EEG-based Motor Imagery classification.

## 4. Discussion

The Kappa value distribution in Figure 4 illustrates the GDC-Net model’s comparative performance against existing MI-EEG classification techniques. The proposed method consistently demonstrates higher agreement with ground truth labels, achieving an average Kappa of 0.7847, outperforming CWT-SCNN (0.657), CNN-MLP (0.569), CapsNet (0.568),

---


## Page 11

Electronics 2025, 14, 2827
&lt;page_number&gt;11 of 14&lt;/page_number&gt;

CutCat (0.4762), and Anchored STFT-Skip-Net (0.520). It also exceeds the reported Kappa values of EEG-Conformer (0.69) and CLT-Net (0.74). Due to the lack of subject-wise Kappa metrics in the original publications of ADFCNN, Conformer, and CLT-Net, a detailed per-subject comparison with these models could not be included. This improvement underscores the effectiveness of the proposed framework in distinguishing MI patterns with more excellent reliability across subjects.

&lt;img&gt;Comparison of Kappa Values Across Subjects for Different Methods&lt;/img&gt;
| Subject | Anchored STFT-Skip-Net | CapsNet | STFT-VGG16 | CutCat | CNN-MLP | CWT-SCNN | Proposed Method |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | 0.60 | 0.58 | 0.45 | 0.50 | 0.48 | 0.48 | 0.80 |
| S2 | 0.15 | 0.20 | 0.21 | 0.12 | 0.29 | 0.62 | 0.56 |
| S3 | 0.12 | 0.10 | 0.34 | 0.21 | 0.43 | 0.10 | 0.49 |
| S4 | 0.91 | 0.94 | 0.85 | 0.95 | 0.92 | 0.97 | 0.97 |
| S5 | 0.75 | 0.66 | 0.62 | 0.66 | 0.59 | 0.85 | 0.93 |
| S6 | 0.67 | 0.67 | 0.72 | 0.50 | 0.49 | 0.73 | 0.81 |
| S7 | 0.51 | 0.51 | 0.42 | 0.50 | 0.42 | 0.74 | 0.83 |
| S8 | 0.74 | 0.83 | 0.43 | 0.83 | 0.76 | 0.69 | 0.89 |
| S9 | 0.74 | 0.71 | 0.42 | 0.86 | 0.69 | 0.42 | 0.86 |
Figure 4. Comparison of Kappa values across subjects for different methods.

Subject-wise analysis reveals that S4, S5, S6, S7, S8, and S9 achieve Kappa values exceeding 0.80, with S4 reaching 0.9712, indicating a near-optimal classification outcome. Even in cases where traditional models show reduced performance, such as S2 and S3, the proposed approach yields substantial improvements, contrasting with the lower scores of Anchored STFT-Skip-Net (0.145, 0.124) and STFT-VGG16 (0.205, 0.338). These findings reinforce the stability and adaptability of the GDC-Net model in handling inter-subject variability.

Table 4 further contextualizes the influence of feature extraction, augmentation strategies, and classifier architectures on classification performance. STFT-based methods (e.g., Anchored STFT-Skip-Net, STFT-VGG16, and CutCat) struggle with time-frequency resolution limitations, impacting their ability to capture critical MI features. CapsNet and CNN-MLP exhibit moderate results, though their limited spatiotemporal modeling capabilities affect classification outcomes. While CWT-SCNN benefits from wavelet-based representation, its predefined mother wavelet selection lacks the adaptability of GMWT. The combination of adaptive feature extraction, deep feature encoding, and sequential modeling in the proposed framework provides a more refined approach to MI classification, improving discrimination between left and right-hand imagery tasks.

Figure 5 depicts the average confusion matrix of the proposed GDC-Net for the BCI IV-2b dataset, showing that accuracies for decoding left and right-hand imagery are 91.17% and 86.56%, respectively. The proportion of misclassifications where imagining left-hand movements was incorrectly identified as right-hand movements reached 8.83%, while instances of imagining right-hand movements being misclassified as left-hand movements stood at 13.44%.

---


## Page 12

Electronics 2025, 14, 2827
&lt;page_number&gt;12 of 14&lt;/page_number&gt;

Table 4. The classification performance and detailed selection, data augmentation, and classification methodologies of different approaches on the BCI Competition IV 2b Dataset.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Data Selection</th>
      <th>Classifier</th>
      <th>Data Augmentation</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CNN-SAE *</td>
      <td>STFT *</td>
      <td>CNN-SAE</td>
      <td>No</td>
      <td>77.60%</td>
    </tr>
    <tr>
      <td>CapsNet *</td>
      <td>STFT</td>
      <td>CNN</td>
      <td>No</td>
      <td>78.44%</td>
    </tr>
    <tr>
      <td>CWT-SCNN *</td>
      <td>CWT *</td>
      <td>SCNN</td>
      <td>No</td>
      <td>83.20%</td>
    </tr>
    <tr>
      <td>CNN-MLP *</td>
      <td>RNN-LSTM</td>
      <td>CNN+MLP</td>
      <td>GAN *</td>
      <td>74.80%</td>
    </tr>
    <tr>
      <td>Skip-NET-GNAA</td>
      <td>Anchored STFT</td>
      <td>Skip-Net-CNN</td>
      <td>GNAA *</td>
      <td>76.00%</td>
    </tr>
    <tr>
      <td>STFT-VGG16</td>
      <td>STFT</td>
      <td>VGG-16</td>
      <td>Cropping</td>
      <td>73.81%</td>
    </tr>
    <tr>
      <td>CutCat</td>
      <td>STFT</td>
      <td>Shallow CNN</td>
      <td>CutCat</td>
      <td>78.44%</td>
    </tr>
    <tr>
      <td>ADFCNN *</td>
      <td>FFT</td>
      <td>ADFCNN</td>
      <td>No</td>
      <td>87.81%</td>
    </tr>
    <tr>
      <td>EEG-Conformer *</td>
      <td>No</td>
      <td>CNN-Conformer</td>
      <td>S&R *</td>
      <td>84.63%</td>
    </tr>
    <tr>
      <td>CLT-Net</td>
      <td>CNN</td>
      <td>LSTM</td>
      <td>S&R *</td>
      <td>87.11%</td>
    </tr>
    <tr>
      <td>Proposed GDC-Net</td>
      <td>GMWT</td>
      <td>CNN-LSTM</td>
      <td>DCGAN</td>
      <td>89.24%</td>
    </tr>
  </tbody>
</table>

* Some acronyms are defined in the following: Stacked Autoencoder: SAE; Short-Time Fourier Transform: STFT; Capsule Network: CapsNet; Simplified Convolutional Neural Network: SCNN; Continuous Wavelet Transform: CWT; Multi-Layer Perception: MLP; Generative Adversarial Networks: GAN; Gradient Norm Adversarial Augmentation: GNAA; ADFCNN: Attention-based dual-scale fusion convolutional neural network; Conformer: Convolutional Transformer; S&R: Segmentation and reconstruction.

&lt;img&gt;A confusion matrix showing the average performance of the proposed GDC-Net. The matrix is divided into four quadrants: Top-left (True Labels: Left Hand, Predicted Labels: Left Hand) with 91.17% in dark blue; Top-right (True Labels: Left Hand, Predicted Labels: Right Hand) with 8.83% in light blue; Bottom-left (True Labels: Right Hand, Predicted Labels: Left Hand) with 13.44% in light blue; Bottom-right (True Labels: Right Hand, Predicted Labels: Right Hand) with 86.56% in dark blue. A color bar on the right ranges from 0 to 100.&lt;/img&gt;
| True Labels | Predicted Left Hand | Predicted Right Hand |
| :--- | :--- | :--- |
| Left Hand | 91.17% | 8.83% |
| Right Hand | 13.44% | 86.56% |

Figure 5. Average confusion matrix of the proposed GDC-Net.

5. Conclusions

This paper introduced GDC-Net, a hybrid framework for MI classification based on EEG signals. The proposed method utilizes GMWT representation for adaptive time-frequency analysis. Within GDC-Net, the CNN component captured spatial features, while LSTM layers modeled temporal dependencies, ensuring an effective representation of MI-related patterns.

Experimental validation on the BCI Competition IV 2b dataset demonstrated the superiority of the GDC-Net, achieving 89.24% classification accuracy in 10-fold cross-validation, outperforming existing methods. Integrating DCGAN for data augmentation significantly improved generalization, mitigating the challenges of limited EEG training data. These findings confirm the effectiveness of combining adaptive wavelet-based feature extraction with deep learning architectures for MI-EEG decoding in BCI applications.

In addition to its classification performance, GDC-Net demonstrates strong computational efficiency and scalability. It achieves an average inference latency of approximately 130 milliseconds per trial and a training time of approximately 485 s per subject. Compared to EEG-Conformer, which required 2000 training epochs and 540 s, and CLT-Net, which

---


## Page 13

Electronics 2025, 14, 2827
&lt;page_number&gt;13 of 14&lt;/page_number&gt;

involved 1000 training rounds, GDC-Net offers a more balanced trade-off between real-time viability, training efficiency, and generalization performance across subjects.

Future research will focus on optimizing computational efficiency for real-time implementation, leveraging transfer learning to enhance cross-subject generalization, and advancing BCI deployment for neurorehabilitation and assistive applications.

**Author Contributions:** Writing—review and editing, Writing—original draft, Software, Methodology, Formal analysis, Conceptualization, Visualization; Software, Validation, Formal analysis, M.B.; Software, Validation, Formal analysis, S.Y.; Conceptualization, Methodology, Investigation, Formal Analysis, Validation, Review, Supervision, W.G. All authors have read and agreed to the published version of the manuscript.

**Funding:** This research was supported by the Hebei Natural Science Foundation (Grant No. A2023205045), the Hebei Central Leading Local Science and Technology Development Foundation (Grant No. 236Z001105G).

**Data Availability Statement:** The data that support the findings of this study are openly available at the following URL/DOI: https://www.bbci.de/competition/iv/ (accessed on 12 March 2024).

**Conflicts of Interest:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

1. Yuan, H.; He, B. Brain–Computer Interfaces Using Sensorimotor Rhythms: Current State and Future Perspectives. *IEEE Trans. Biomed. Eng.* **2014**, *61*, 1425–1435. [CrossRef] [PubMed]
2. Phothisonothai, M.; Nakagawa, M. EEG-Based Classification of Motor Imagery Tasks Using Fractal Dimension and Neural Network for Brain-Computer Interface. *IEICE Trans. Inf. Syst.* **2008**, *91*, 44–53. [CrossRef]
3. Zhu, X.; Li, P.; Li, C.; Yao, D.; Zhang, R.; Xu, P. Separated Channel Convolutional Neural Network to Realize the Training Free Motor Imagery BCI Systems. *Biomed. Signal Process. Control* **2019**, *49*, 396–403. [CrossRef]
4. Sakhavi, S.; Member, S.; Guan, C. Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks. In Proceedings of the 2018 6th International Conference on Brain-Computer Interface, Jeju, Republic of Korea, 15–17 January 2018; pp. 5619–5629.
5. Tabar, Y.R.; Halici, U. A Novel Deep Learning Approach for Classification of EEG Motor Imagery Signals. *J. Neural Eng.* **2017**, *14*, 16003. [CrossRef] [PubMed]
6. Zhao, D.; Tang, F.; Si, B.; Feng, X. Learning Joint Space–Time–Frequency Features for EEG Decoding on Small Labeled Data. *Neural Netw.* **2019**, *114*, 67–77. [CrossRef]
7. Salamon, J.; Bello, J.P. Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Process. Lett.* **2017**, *24*, 279–283. [CrossRef]
8. Pérez-Benítez, J.L.; Pérez-Benítez, J.A.; Espina-Hernández, J.H. Development of a Brain Computer Interface Interface Using Multi-Frequency Visual Stimulation and Deep Neural Networks. In Proceedings of the 2018 International Conference on Electronics, Communications and Computers (CONIELECOMP), Cholula, Mexico, 21–23 February 2018; IEEE: Piscataway, NJ, USA, 2018; pp. 18–24.
9. Li, H.; Zhang, D.; Xie, J. MI-DABAN: A Dual-Attention-Based Adversarial Network for Motor Imagery Classification. *Comput. Biol. Med.* **2023**, *152*, 106420. [CrossRef]
10. Roy, S.; Dora, S.; McCreadie, K.; Prasad, G. MIEEG-GAN: Generating Artificial Motor Imagery Electroencephalography Signals. In Proceedings of the 2020 International Joint Conference on Neural Networks (IJCNN), Glasgow, UK, 19–24 July 2020; pp. 1–8.
11. Yang, L.; Song, Y.; Ma, K.; Xie, L. Motor Imagery EEG Decoding Method Based on a Discriminative Feature Learning Strategy. *IEEE Trans. Neural Syst. Rehabil. Eng.* **2021**, *29*, 368–379. [CrossRef]
12. Habashi, A.G.; Azab, A.M.; Eldawlatly, S.; Gamal, M.A. Generative Adversarial Networks in EEG Analysis: An Overview. *J. Neuroeng. Rehabil.* **2023**, *20*, 40. [CrossRef]
13. Song, Y.; Yang, L.; Jia, X.; Xie, L. Common Spatial Generative Adversarial Networks Based EEG Data Augmentation for Cross-Subject Brain-Computer Interface. *arXiv* **2021**, arXiv:2102.04456.
14. Leeb, R.; Brunner, C. *BCI Competition 2008–Graz Data Set B*.; Graz University of Technology: Graz, Austria, 2008; Volume 16, pp. 1–6.
15. Chung, J.W.; Ofori, E.; Misra, G.; Hess, C.W.; Vaillancourt, D.E. Beta-Band Activity and Connectivity in Sensorimotor and Parietal Cortex Are Important for Accurate Motor Performance. *Neuroimage* **2017**, *144*, 164–173. [CrossRef] [PubMed]

---


## Page 14

Electronics 2025, 14, 2827
&lt;page_number&gt;14 of 14&lt;/page_number&gt;

16. Olhede, S.C.; Walden, A.T. Generalized Morse Wavelets. *IEEE Trans. Signal Process.* **2002**, *50*, 2661–2670. [CrossRef]
17. Lilly, J.M. Element Analysis: A Wavelet-Based Method for Analysing Time-Localized Events in Noisy Time Series. *Proc. R. Soc. A Math. Phys. Eng. Sci.* **2017**, *473*, 20160776. [CrossRef] [PubMed]
18. Martinez-Ríos, E.A.; Bustamante-Bello, R.; Navarro-Tuch, S.; Perez-Meana, H. Applications of the Generalized Morse Wavelets: A Review. *IEEE Access* **2022**, *11*, 667–688. [CrossRef]
19. Radford, A.; Metz, L.; Chintala, S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *arXiv* **2015**, arXiv:1511.06434.
20. Cubuk, E.D.; Zoph, B.; Mane, D.; Vasudevan, V.; Le, Q.V. Autoaugment: Learning Augmentation Policies from Data. *arXiv* **2018**, arXiv:1805.09501.
21. Ali, O.; Saif--ur--Rehman, M.; Dyck, S.; Glasmachers, T.; Iossifidis, I.; Klaes, C. Enhancing the Decoding Accuracy of EEG Signals by the Introduction of Anchored -- STFT and Adversarial Data Augmentation Method. *Sci. Rep.* **2022**, *12*, 4245. [CrossRef]
22. Ha, K.W.; Jeong, J.W. Motor Imagery EEG Classification Using Capsule Networks. *Sensors* **2019**, *19*, 2854. [CrossRef]
23. Xu, G.; Shen, X.; Chen, S.; Zong, Y.; Zhang, C.; Yue, H.; Liu, M.; Chen, F.; Che, W. A Deep Transfer Convolutional Neural Network Framework for EEG Signal Classification. *IEEE Access* **2019**, *7*, 112767–112776. [CrossRef]
24. Al-Saegh, A.; Dawwd, S.A.; Abdul-Jabbar, J.M. CutCat: An Augmentation Method for EEG Classification. *Neural Netw.* **2021**, *141*, 433–443. [CrossRef]
25. Yang, B.; Fan, C.; Guan, C.; Gu, X.; Zheng, M. A Framework on Optimization Strategy for EEG Motor Imagery Recognition. In Proceedings of the 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Berlin, Germany, 23–27 July 2019; pp. 774–777.
26. Li, F.; He, F.; Wang, F.; Zhang, D.; Xia, Y.; Li, X. A Novel Simplified Convolutional Neural Network Classification Algorithm of Motor Imagery EEG Signals Based on Deep Learning. *Appl. Sci.* **2020**, *10*, 1605. [CrossRef]
27. Tao, W.; Wang, Z.; Wong, C.M.; Jia, Z.; Li, C.; Chen, X.; Chen, C.L.P.; Wan, F. ADFCNN: Attention-Based Dual-Scale Fusion Convolutional Neural Network for Motor Imagery Brain-Computer Interface. *IEEE Trans. Neural Syst. Rehabil. Eng.* **2024**, *32*, 154–165. [CrossRef] [PubMed]
28. Song, Y.; Zheng, Q.; Liu, B.; Gao, X. EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. *IEEE Trans. Neural Syst. Rehabil. Eng.* **2023**, *31*, 710–719. [CrossRef]
29. Gu, H.; Chen, T.; Ma, X.; Zhang, M.; Sun, Y.; Zhao, J. CLTNet: A Hybrid Deep Learning Model for Motor Imagery Classification. *Brain Sci.* **2025**, *15*, 124. [CrossRef] [PubMed]

**Disclaimer/Publisher's Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.