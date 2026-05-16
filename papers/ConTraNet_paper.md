

---

## Page 1

Computers in Biology and Medicine 168 (2024) 107649
Available online 2 November 2023
0010-4825/© 2023 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).
ConTraNet: A hybrid network for improving the classification of EEG and
EMG signals with limited training data
Omair Ali a,d,*, Muhammad Saif-ur-Rehman c, Tobias Glasmachers b, Ioannis Iossifidis c,
Christian Klaes a
a Faculty of Medicine, Department of Neurosurgery, University Hospital Knappschaftskrankenhaus Bochum GmbH, Germany
b Institut für Neuroinformatik, Ruhr University Bochum, Germany
c Department of Computer Science, Ruhr-West University of Applied Science, Mülheim an der Ruhr, Germany
d Department of Electrical Engineering and Information Technology, Ruhr-University Bochum, Germany

## A R T I C L E  I N F O

Keywords:
Brain computer interface
EEG decoding
EMG decoding
Machine learning
Deep learning

## A B S T R A C T

Objective: Bio-Signals such as electroencephalography (EEG) and electromyography (EMG) are widely used for
the rehabilitation of physically disabled people and for the characterization of cognitive impairments. Successful
decoding of these bio-signals is however non-trivial because of the time-varying and non-stationary character­
istics. Furthermore, existence of short- and long-range dependencies in these time-series signal makes the
decoding even more challenging. State-of-the-art studies proposed Convolutional Neural Networks (CNNs) based
architectures for the classification of these bio-signals, which are proven useful to learn spatial representations.
However, CNNs because of the fixed size convolutional kernels and shared weights pay only uniform attention
and are also suboptimal in learning short-long term dependencies, simultaneously, which could be pivotal in
decoding EEG and EMG signals. Therefore, it is important to address these limitations of CNNs. To learn short-
and long-range dependencies simultaneously and to pay more attention to more relevant part of the input signal,
Transformer neural network-based architectures can play a significant role. Nonetheless, it requires a large
corpus of training data. However, EEG and EMG decoding studies produce limited amount of the data. Therefore,
using standalone transformers neural networks produce ordinary results. In this study, we ask a question whether
we can fix the limitations of CNN and transformer neural networks and provide a robust and generalized model
that can simultaneously learn spatial patterns, long-short term dependencies, pay variable amount of attention to
time-varying non-stationary input signal with limited training data.
Approach: In this work, we introduce a novel single hybrid model called ConTraNet, which is based on CNN and
Transformer architectures that contains the strengths of both CNN and Transformer neural networks. ConTraNet
uses a CNN block to introduce inductive bias in the model and learn local dependencies, whereas the Transformer
block uses the self-attention mechanism to learn the short- and long-range or global dependencies in the signal
and learn to pay different attention to different parts of the signals.
Main results: We evaluated and compared the ConTraNet with state-of-the-art methods on four publicly available
datasets (BCI Competition IV dataset 2b, Physionet MI-EEG dataset, Mendeley sEMG dataset, Mendeley sEMG V1
dataset) which belong to EEG-HMI and EMG-HMI paradigms. ConTraNet outperformed its counterparts in all the
different category tasks (2-class, 3-class, 4-class, 7-class, and 10-class decoding tasks).
Significance: With limited training data ConTraNet significantly improves classification performance on four
publicly available datasets for 2, 3, 4, 7, and 10-classes compared to its counterparts.
* Corresponding author. Faculty of Medicine, Department of Neurosurgery, University Hospital Knappschaftskrankenhaus Bochum GmbH, Germany.
E-mail address: omair.ali@ruhr-uni-bochum.de (O. Ali).
Contents lists available at ScienceDirect
Computers in Biology and Medicine
journal homepage: www.elsevier.com/locate/compbiomed
https://doi.org/10.1016/j.compbiomed.2023.107649
Received 30 June 2023; Received in revised form 6 October 2023; Accepted 31 October 2023


---

## Page 2

Computers in Biology and Medicine 168 (2024) 107649
2
1. Introduction
1.1. Human machine interfaces (HMI): non-invasive brain computer
interfaces (BCI) and electromyography-based HMI (EMG-HMI)
A Human Machine Interface (HMI) is a framework that aims to
minimize disability of the patients by providing a connection and con­
trol between a user and an external machine useful for rehabilitation
[1]. Two widely used HMI paradigms are brain computer interface (BCI)
[2,3] and electromyography-based HMI (EMG-HMI) [4,5]. In BCI par­
adigms, brain signals of the user’s intent are translated into corre­
sponding command signals to control an external device [6], whereas in
EMG-HMI paradigm, muscle movements are used [7].
Both paradigms generally consist of five processing steps [5,8]: data
recording, pre-processing, feature extraction, inference, and control. In
both HMI types, the data is recorded either by invasive or non-invasive
methods [9–14]. Non-invasive BCI and EMG-HMI systems have gained
more traction compared to invasive approaches in medical applications
because of their convenient, low risk and non-complex nature [15–19].
EEG signals are the primary source of recording electrical brain activity
recorded from electrodes which are placed on the scalp. Similarly, sur­
face EMG (sEMG) signals are one of the most common used signals in
non-invasive EMG-HMI systems. sEMG signals are the electrical activity
generated during muscle contractions which is recorded using elec­
trodes placed on the skin.
EEG based BCI systems are often used to decode motor imagery (MI-
EEG) tasks by recording from the brain areas which process motor
related activities. Well researched area for processing motor related
tasks is primary motor cortex (M1) and posterior parietal cortex (PPC)
[10].
1.2. Existing state-of-art algorithms for decoding MI-EEG and sEMG
signals
Convolutional neural network (CNN): CNN based architectures
have shown leading results in the classification of MI-EEG and sEMG
signals as well as in other domains of BCI [20–24].
Strength of CNN: One of the reasons for the successful implication of
CNN based models in HMI paradigms is the built-in hard inductive bias
such as translation invariance and translation equivariance, in the ar­
chitectures which is very useful for smaller datasets [25]. Inductive bias
can be defined as the prioritization of some hypotheses in hypothesis
space to extract information from the input data. Translation invariance
bias in CNN refers to the ability of CNN based architectures to extract the
features from the input irrespective of their location in the input. In time
series signals such as MI-EEG and sEMG signals which are non-stationary
in nature, and where the time location of relevant features is random
and unknown, translation invariance offered by CNN plays a vital role in
their extraction.
Drawback of CNN based architectures: Since, MI-EEG and sEMG
are time series signals which have strong 1D local structure: variables
that are spatially or temporally nearby are highly correlated [25], hence
it is pivotal to learn and extract local and deep features from the signals.
On the other hand, MI-EEG and sEMG are time-series signals, where
interconnection of different time samples could represent user intention
[26]. Therefore, it is also of vital importance to learn the long-range
dependencies between different time samples to correctly decode
them. However, CNN based models present a drawback in their archi­
tecture regarding this, that is: the feature extractors based on CNN ar­
chitecture considerably rely on the selection of kernel sizes [27–29].
CNN architectures need big kernels to learn long-range dependencies
which in turn restricts learning small, local, and deep features. Alter­
natively, the depth of the architecture can be enhanced using small size
kernels in each layer such that the early layers can extract the local and
small size features whereas, the later layers have bigger receptive fields
which cover the whole signal for learning larger features. This approach
is however computationally expensive and slow and require more
training data to train deeper architecture and still does not guarantee the
extraction of global context because of the long trace of transformed
input through hidden layers [25]. Additionally, since each convolutional
layer of a CNN architecture shares weights and is unable to learn glob­
ally available context in the given input, which means that it uniformly
attends to each part of the MI-EEG and sEMG signal in that layer to learn
meaningful patterns. Whereas MI-EEG and sEMG signals are
non-stationary in nature henceforth, each part of the signal might not be
equally significant for correctly classifying them. Therefore, more
attention should be paid to the relevant parts of the signal compared to
less relevant parts.
Transformer based architectures: To learn long-distance features
and to attend specific parts of the signals, a new architecture based on
self-attention mechanism called, the Transformer was introduced in
Ref. [30]. The transformer architecture was originally introduced for
natural language processing (NLP), but it has found its way to many
other applications due to its effectiveness in learning long-range global
dependencies and ability to parallelize the computation [31,32].
Drawback of Transformer based architectures: Transformer ar­
chitecture works equally well on short as well as long sequences.
However, the direct applicability of Transformers in classifying MI-EEG
and sEMG signals remains limited. This is because, transformers lack the
inductive bias in their design, which means that the transformer-based
architectures do not prioritize any hypotheses in hypothesis space by
their design to extract the information from the input. Rather, they are
more flexible and can extract the rules from the data without having any
prior bias to generalize on the unseen data. It implies that all the rules
from the data which are required for better generalization must be
learned from scratch which usually requires a very large amount of data
to pretrain them and then transfer to mid-sized recognition tasks [32]. It
also implies that transformer-based architectures perform worse in low
data setting. Since the data scarcity issue prevails in EEG and sEMG
based HMI paradigms, hence it limits the direct and stand-alone use of
transformers in this field.
1.3. Proposed solution
In this study, we address the following limitations posed by stand­
alone CNN and transformer-based architectures employed for neuro­
physiological signals.
1. Limitations of learning local and global context simultaneously from
the signals due to fixed kernel size in CNN architectures.
2. Limitations of learning to pay more attention to the part of the signal
crucial for its correct decoding due to weight sharing in CNN
architectures.
3. Limitations of learning the inductive bias from scratch for successful
decoding of the signals using limited training data in transformer
architectures.
4. Limitations of attending to relevant part of the signal crucial for
accurate classification of the user’s intent using limited training data
in transformer architectures.
To address the limitations described above, we introduced a hybrid
model, named ConTraNet, that combines the strengths of both CNN and
transformer architectures. ConTraNet consists of a convolutional block
(CNN block) and a transformer block and addresses these limitations as.
1. CNN block in ConTraNet learns the local dependencies whereas the
Transformer block learns the long-range global dependencies
simultaneously from the signal.
2. CNN block introduces the inductive bias such as translation invari­
ance in the architecture whereas Transformer block learns to pay
more attention to the relevant parts of the signal crucial for correct
decoding. It also suppresses the less relevant parts.
O. Ali et al.


---

## Page 3

Computers in Biology and Medicine 168 (2024) 107649
3
3. Inclusion of inductive bias and ability to learn local dependencies by
CNN block in the architecture, ensures that transformer block now
requires less amount of training data to the extract long-range de­
pendencies and perform part specific attention on the signal.
The contribution of this study is to address the limitations put for­
ward by stand-alone CNN and transformer-based architectures for
neurophysiological signals. Moreover, the motivation behind this study
is to design a compact architecture which is suitable for decoding both
EEG and sEMG signals simultaneously and is also suitable to be deployed
on
wearable edge
devices such
as neuromorphic chip for
neurorehabilitation.
In addition, the focus is to show the significance of our hybrid ar­
chitecture based ConTraNet in comparison to standalone CNN or
transformer neural network-based architectures for decoding MI-EEG
signals and sEMG signals with respect to following perspectives.
1. Generalization performance comparison.
2. Amount of training data required.
Experimental evaluation during this study established that ConTra­
Net outperformed its counterparts in all above-mentioned criteria on
four publicly available MI-EEG and sEMG datasets.
The rest of the manuscript is organized as follows: Section 2.1 ex­
plains the datasets used during this study to investigate the proposed
approach. Section 2.2 introduces the architectural details of the pro­
posed method ConTraNet. Section 3 presents the results of this study,
where we first address the limitations of learning short as well as global
dependencies (section 3.1). Secondly, in section 3.2, we present the
ability of the model to train on limited amount of training data
compared to other models. Thirdly, in section 3.3, we present the
generalization ability of the ConTraNet on four versatile publicly
available datasets. Section 4 presents the Discussion and Summary while
the conclusion is presented in section 5.
2. Materials and Methods
2.1. Data description
Two different physiological signals (MI-EEG signals and sEMG) and
four publicly available benchmark datasets are used to quantify and
validate the high generalization property, requirement of less training
data, and high evaluation performance of ConTraNet in comparison to
existing state of the art algorithms. High variability in the given datasets
is necessary for the evaluation of the algorithms from different angles
and perspectives. We selected a benchmark dataset which contains the
data of more than 100 subjects with four different MI-Tasks [33].
Furthermore, another benchmark dataset from 40 subjects with 10 hand
gestures [34] to ensure high variability in the given datasets. Moreover,
another benchmark dataset which contains data of 30 subjects with 7
different hand tasks [35]. In addition, another benchmark dataset [36]
comprising the data of 9 subjects with two different MI-tasks is used.
Highly used benchmark datasets are the main criterions for the selection
of the datasets for both EEG and sEMG tasks. A brief but concise
explanation of all the datasets is provided in the following sections.
2.1.1. BCI competition IV dataset 2b (MI-EEG)
The dataset consists of two classes of MI-EEG signals of the left (L)
and right hand (R) movement imagination of 9 subjects with normal or
corrected-to-normal vision. Each subject performed five sessions (01T,
02T, 03T, 04E and 05E). The MI-EEG data is recorded using three bi­
polar electrodes (C3, Cz and C4), sampled at 250 Hz and bandpass
filtered between 0.5 Hz and 100 Hz. A notch filter at 50 Hz is applied.
For training and testing, the MI-EEG signal from second 3 to second 5.5
(2.5s in total) containing motor imagination of left or right hand based
on the cue is considered for each trial. The timing window used here is
the same as used in Ref. [22]. We perform 2-class (L/R) classification on
this dataset. The detailed description of the dataset is available in
Ref. [36].
2.1.2. Physionet MI-EEG dataset
The dataset contains MI/ME-EEG recordings of 109 healthy subjects.
Four subjects are dropped from the dataset since they did perform fewer
trials than the rest, resulting in 105 subjects to be used in this study.
Subjects performed both ME and MI-tasks, but we just focused on the
classification of MI-tasks in this work. MI-EEG signals are recorded with
the BCI-2000 system using 64 electrodes, sampled at 160 Hz. Each
subject performed three runs of the MI-task: left fist (L) or right fist (R)
movement imagination. Additionally, they also performed three runs of
the MI-task for both fists (B) or both feet (F) movement imagination.
Moreover, the baseline run provides the resting data (0) where the
subjects did not receive any cues. Each run is 120s long and contains 14
MI-trials, resulting in 42 trials (21 trials per class) per subject. For this
dataset, we distinguish between 2-class (L/R), 3-class (L/R/0) and 4-
class (L/R/0/F) MI-tasks. A detailed description of the dataset is avail­
able in Ref. [33].
2.1.3. Mendeley Data - sEMG
The dataset contains the 4-channel sEMG recordings of 10 different
hand gestures from 40 healthy subjects. The data are recorded with a
BIOPAC MP36 device using Ag/AgCl surface bipolar electrodes, sampled
at 2 kHz and bandpass filtered between 5 Hz and 500 Hz using a sixth-
order Butterworth bandpass filter and a notch filter is applied at 50 Hz.
Each subject participated in five repetitive cycles with 30s rest in be­
tween cycles. All cycles contain the 10 hand gestures. Each cycle lasts for
104s, where each hand gesture is 6s long with 4s resting in between two
hand gestures. The hand gestures performed by the subjects are rest or
neutral state, extension of the wrist, flexion of the wrist, ulnar deviation
of the wrist, radial deviation of the wrist, grip, abduction of all fingers,
adduction of all fingers, supination, and protonation. For this dataset,
we distinguish between 10-class sEMG signals. The detailed explanation
and description of the data is presented in Ref. [34].
2.1.4. Mendeley Data – sEMG V1
The dataset contains the 4-channel sEMG recordings of 7 different
hand gestures from 30 healthy subjects. The data are recorded with a
BIOPAC MP36 device using Ag/AgCl surface bipolar electrodes, sampled
at 2 kHz and bandpass filtered between 5 Hz and 500 Hz using a sixth-
order Butterworth bandpass filter and a notch filter is applied at 50 Hz.
Each subject participated in five repetitive cycles with 30s rest in be­
tween cycles. All cycles contain the 7 hand gestures. Each cycle lasts for
74s, where each hand gesture is 6s long with 4s resting in between two
hand gestures. The hand gestures performed by the subjects are rest or
neutral state, extension of the wrist, flexion of the wrist, ulnar deviation
of the wrist, radial deviation of the wrist, punch, and open hand. For this
dataset, we distinguish between 7-class sEMG signals. The detailed
explanation and description of the data is presented in Ref. [35].
2.2. Methods
Here, we introduce ConTraNet, an end-to-end CNN-Transformer
based hybrid architecture. A brief explanation of CNN and Transformer
architectures is presented in section ‘Appendix’ of the manuscript.
2.2.1. ConTraNet
ConTraNet as shown in Fig. 1 is a hybrid architecture which consists
of CNN and the Transformer block.
CNN block: CNN block induces inductive biases such as translation
equivariance and locality in the architecture which is specifically
important for learning meaningful patterns from a small-size datasets.
Additionally, it also learns or extracts the temporal features from the
signals. Apart from this, it also learns the local dependencies within the
O. Ali et al.


---

## Page 4

Computers in Biology and Medicine 168 (2024) 107649
4
Fig. 1. ConTraNet. The model architecture. It has two blocks: a CNN block and a Transformer block. Input is first fed to the CNN block which generates the feature
maps at the output which are the input to the Transformer block. The output of the Transformer encoder is fed to the MLP head for final classification.
O. Ali et al.


---

## Page 5

Computers in Biology and Medicine 168 (2024) 107649
5
kernel length which are important for online and offline decoding.
Another perspective of the CNN block is that it guides the Transformer
block by learning to extract the relevant local temporal features and
forwards them to the Transformer block to find the features and global
dependencies among them.
The CNN block in ConTraNet contains a convolutional layer, pooling
layer, regularization layer, a normalization layer, and an activation
layer. The convolutional layer uses convolutional filters of size (1,
sampling frequency/2) to acquire the frequency resolution of 2 Hz for
EEG signals. Henceforth it allows the model to capture frequency in­
formation at 2 Hz and above. Batch normalization is then applied to the
feature maps followed by exponential linear unit (ELU) non-linearity
[37], which generates the smooth response for negative inputs
compared to the transient response of RELU for negative inputs. Average
pooling layer of size (1,8) is used to reduce the sampling rate of the
signal to 1/8th of the sampling frequency/2. To regularize the model
and prevent it from overfitting, dropout technique [38] with dropout
rate of 0.5 is used in CNN block. Spatial dropout is used for MI-EEG
signals instead of normal dropout similar to Ref. [23] as it works
slightly better for MI-EEG signals, whereas normal dropout is used for
sEMG signals. Additionally, each temporal filter is regularized by using a
maximum norm constraint of 0.25 on its weights to avoid overfitting.
The inputs to the CNN block are of the 2-dimensions which are given
as Nch x Ns; where Nch = number of channels and Ns = number of
samples. However, to be compatible with Keras in Python, the inputs are
reshaped to B x Nch x Ns x 1; where B = batch size.
Transformer block: Transformer block learns the global or long-
range dependencies, which play a vital role in successful decoding of
MI-EEG and sEMG signals. Additionally, it learns to pay more attention
to the relevant parts of the signals which are vital for the correct clas­
sification while suppressing the non-relevant parts.
The Transformer block in ConTraNet contains only the encoder fol­
lowed by classification head (multi-layer perceptron (MLP) head and a
softmax layer). The output of the CNN block is the input of the Trans­
former block and has image like structure. Since the Transformer model
was originally used for NLP tasks and only accepts the embeddings of the
sequence tokens as its inputs, henceforth could not be directly used for
images. However [32] presented an approach to transform an image into
sequence of linear embeddings which could be used as inputs to the
Transformer model. Inspired by this approach, the feature maps of CNN
block are transformed into respective embeddings which are provided as
input to the Transformer block.
The CNN block outputs the feature maps of shape [Fch x Fs x Nf],
where Fch = height, Fs = width and, Nf = number of filters. Firstly, image
patches of shape [Ph x Pw x Nf] are extracted from the feature maps using
patch stride 1. Where, Ph = Fch and Pw = 2. The image patches are
overlapped by factor of Fs = 1.
Here Fch is computed by following equation.
Fch = Nch −pool size height
stride height
+ 1
Where, pool_size_height = 1 and stride_height = 1. In above equation,
pool_size_height and stride_height are two hyperparameters of average
pooling layer in CNN block. Their values are set to 1 to reduce the
sampling rate of individual channels respectively instead of multiple
channels collectively. The sampling rate is reduced to decrease the
computational cost as well as to avoid overfitting in the model. There­
fore, using the equation above the height of patches results into number
of channels in the input. Whereas the value of Pw = 2 is a hyper­
parameter and is computed using grid search method. The impact of
different patch width on the performance of ConTraNet is presented in
Fig. 11.
The extracted patches are linearly projected to obtain the learnable
patch embeddings which are then added with learnable positional
encodings to include the position information. The embedded patches
(patch embeddings + positional embeddings) are sent to the Trans­
former Encoder as its input.
The Transformer encoder in ConTraNet consists of a single encoder
layer. The encoder layer contains multi-headed self-attention and a MLP
block. The MLP block contains two hidden layers with an ELU non-
linearity. Layer normalization is applied before MLP and attention
blocks, whereas residual connection is applied after MLP and attention
blocks. The Transformer encoder is followed by a classification head
which is implemented by another MLP with one hidden layer and a
softmax layer. Dropout of 0.7 is used in MLP block of encoder layer as
well as classification head to regularize the model and prevent over­
fitting. Additionally, each hidden layer is regularized by using a
maximum norm constraint of 0.25 on its weights to avoid overfitting.
Since the Transformer uses constant latent vector size dmodel, also
referred as model dimension, through all its layers, henceforth, all the
transformations (patch embeddings, positional embeddings, output of
multi-head self-attention, output of MLP block in encoder layer) applied
match the model dimension.
Transformer block in ConTraNet, uses only the transformer encoder
part of the vanilla transformer architecture presented in Ref. [30].
Transformer architecture is originally introduced by Ref. [30] for
NLP tasks such as machine translations and has since been used in many
other NLP tasks such as text generation. The original transformer model
has an encoder-decoder architecture. The encoder part of the trans­
former architecture maps the inputs to the latent space while the
decoder part uses that latent representation and maps it to the required
output in autoregressive manner i.e., outputting single output (word in
case of translation and text generation tasks) at a time and then
combining the output with the input to produce next word. The
encoder-decoder configuration of the transformer is usually used for
generative tasks such as text generation or machine translation among
others.
However, in our case, the goal is to classify the input data rather than
generating another sequential output, therefore, in addition to learning
input embeddings and positional encodings, we only employed the
transformer encoder part which maps an input sequence into sequence
of latent representations of the input which are finally used by a multi-
layer perceptron (MLP) head to generate the final classification results.
Workflow of transformer block in ConTraNet: The outputs
(feature maps) of the CNN block are the inputs to the transformer block,
where these are processed as follows:
1. The patches are extracted from the feature maps.
2. These patches are then flattened and linearly projected using a
learnable linear projection to get their embeddings.
3. Then learnable position embeddings are computed and added
together with patch embeddings to encode the position information
in the input.
4. Afterwards these embedded patches (patch embedding + positional
encoding) are fed to the transformer encoder layer as input, where
they are normalized.
5. Then these normalized embedded patches are used by the multi head
attention layer, to compute the attention score of each of these
patches based on their importance.
6. Afterwards, the weighted output of the multi head attention layer
and the embedded patches from step 3 are added together.
7. Then this joint representation is normalized and fed to point wise
feed forward network (MLP).
8. The output of this feed forward network is added together with
output from step 6 using a skip connection.
9. Finally, this joint representation is sent forth to a classification head
(MLP and softmax) which outputs the final classification result.
Reasons for selection of the architectural design: The main
motivation behind designing the hybrid ConTraNet model is to address
the limitations of stand-alone CNN and transformer architectures
O. Ali et al.


---

## Page 6

Computers in Biology and Medicine 168 (2024) 107649
6
employed for neurophysiological signals. The inclusion of CNN block
introduces the inductive bias in the model which addresses the limita­
tion of requiring large corpus of data for the training of stand-alone
transformer architecture. Whereas the inclusion of self-attention
mechanism in transformer block addresses the inability of CNN block
to learn long range dependencies due to its small and fixed size kernels
which are essential for the decoding of neurophysiological signals like
EEG and EMG.
The blocks in ConTraNet are designed in modular fashion, which
implies that each block could encapsulate more layers of its kind,
ensuring the scaling up and down capability of the model, without
affecting the overall architecture design of ConTraNet. However, there
are following constraints that should be taken care of while designing an
architecture which is suitable for deploying in real time neuro­
rehabilitation processes:
1. Limited amount of available data: Lack of available data in real
time scenarios is a primary source of concern in the success of
automated neurorehabilitation process. Therefore, one of the moti­
vations of this study is to design an architecture which is capable of
learning to decode EEG and EMG signals using limited amount of
data without compromising its statistical efficiency.
2. Limited amount of memory and computational power offered
by embedded hardware: Another source of concern is the compu­
tational power available to use deep learning models in real time
scenarios on edge devices such as neuromorphic chips.
Keeping these constraints in sight, in this study we proposed Con­
TraNet (an end-to-end deep learning model) as small and compact as
possible without compromising its statistical efficiency, by using only
single convolutional and transformer encoder layer (ConTraNet1,16:
ConTraNet model with 16 convolutional kernels and 1 encoder layer).
Adding more convolutional layers adds more computational cost since
the convolutional layers are one of the most computational expensive
layers in deep learning models [30]. Secondly, it adds more complexity
to the model by adding more training parameters. Similarly, adding
more encoder layers in the transformer block will cause the model to
overfit in limited data settings.
Henceforth, to avoid making model more complex and deeper, which
also requires more computational resources and may result into over­
fitting in the case of limited training data and consequently affects the
generalization capabilities of the model, we used ConTraNet1,16 for the
analysis of EEG and EMG.
However, the convolutional kernels of the CNN block, and patch
width of the transformer block are among the hyperparameters which
are obtained after using grid search method. The impact of different
convolutional kernels and different patch width on the performance of
ConTraNet are presented in Table 12 and Fig. 11.
The hyperparameters used for the training of ConTraNet after tuning
are shown in Table 1.
3. Results
Here, a comprehensive performance evaluation and comparison of
the proposed method with existing state-of-the-art methods is presented
in three contexts. First, we emphasized on learning the important local
and global context for time-series signals like EEG and sEMG. In the
second analysis, we showed that ConTraNet requires less training data in
comparison to state-of-art standalone CNN and transformer-based al­
gorithms. Then, we evaluated the generalization quality of ConTraNet
on the datasets with high variability.
3.1. Extracting local and global context
Physiological signals like EEG and sEMG are time-varying and non-
stationary time-series signals. It implies that the information carried
by these signals is non-static and can be interlinked at the same time. By
interlinked, we mean that two instances occurring at farthest time points
could be related to each other and form a collective decision about the
existence of the whole signal. While it is crucial to learn and extract the
local information from the signal, it is equally important and significant
to extract global context to give meaning to the entire signal and
correctly decode it. On the same hand, it is notable to put more weight or
emphasis on the relevant part of the signal and subdue the impertinent
part of the signal to take out the correct global context.
Here, we present the local and global context learned and extracted
by ConTraNet during its training on dataset 2b from BCI Competition IV.
To exhibit that, we analyzed the attention maps of the Transformer
block of ConTraNet. Transformer block learns the local and global
context using the filtered signals provided to it as input by the CNN
block. During training it learns to pay more attention to the parts of the
signal which play vital role in the correct decoding of the signals and
suppresses the non-relevant parts of signal. Each head in multi-head
attention layer of the transformer block operates on the entire signal
and decides which part of the signal contains more relevant information
and extracts correspondingly different patterns from the signal, hence­
forth paying different attention to different parts of the signal, which is
vital for extracting meaningful features from non-stationary and noisy
time-series signal like EEG and EMG. The information extracted by all
heads of the multi-head attention layer is used to learn the local and
global context during the training.
Here, ConTraNet is first trained on the entire dataset and then it is
employed to acquire the attention maps of an input signal which is
shown in Fig. 2 (a). The mean attention map computed by calculating
the mean of all attention heads is shown in Fig. 2 (b). The intensity of the
color in the plot depicts the attention paid by ConTraNet to the
respective part of the signal. It not only shows how a sample in each
attention head is attending to its neighborhood but also to the samples
farthest away and hence collecting both local and global context. The
local and global context from all the attention heads is used to formulate
the final decision about the signal. It is shown in Fig. 2 (a), that each
head pays different attention to different parts of the input signal.
Attention head 0 pays more attention to the first half of the signal
compared to Attention head 2 which not only pays more attention to the
start of the signal, but also some attention towards the first half as well
as the end of the signal. Fig. 2 (b) shows the mean attention map of the
attention heads. It is shown here that more attention is paid to start, and
first half of the signal compared to second half and the end of the signal
to extract relevant local and global dependencies from this input to
correctly decode it. Fig. 2 (b) also contains the EEG signal shown below
the mean attention head. The projection of the mean attention on this
EEG signal shows that ConTraNet paid more attention around the peaks
shown in the EEG signal around 10–50-time samples as well as around
Table 1
Hyperparameters of vanilla ConTraNet.
Hyperparameters
Values
Optimization algorithm
Adam
Epochs
100
Learning rate
0.001 (epochs 0–50), 0.0001 (epochs 51–100)
No. of convolutional kernels
16
Padding in Convolutional layer
Same
dmodel
32
No. of heads
8
Pw, Ph
2, Fch
Stride in Patch extractor
1
Padding in Patch extractor
Valid
Hidden units in Encoder MLP block
2* dmodel , dmodel
Hidden units in MLP Head
112
Filter size MI-EEG
(1, sampling frequency/2)
Pool size
(1,8)
Dropout in CNN block
0.5
Dropout in MLP block
0.7
O. Ali et al.


---

## Page 7

Computers in Biology and Medicine 168 (2024) 107649
7
150-180-time samples whereas it suppressed the signal around the dips
such as around time samples 180–220 and 550–570.
3.2. Training using limited amount of data
Here, we show the ability of the ConTraNet to converge faster than
standalone Transformer and state-of-the art CNN based architectures
[23,24] using different configuration for training data selection.
In the first analysis, we defined a purely transformer-based archi­
tecture called TraNet which has only one encoder layer. We kept only
one encoder layer to match the transformer block of vanilla ConTraNet.
This keeps TraNet as shallow as possible which ensures the minimum
amount of training parameters and helps avoid overfitting.
BCI competition IV, dataset 2b is used for this analysis. Both the
models were trained on the same training examples (4 randomly
selected subject’s data with almost 400 examples each) with same
hyperparameters settings (see Table 1). Fig. 3 shows the comparison of
training and validation loss computed on the unseen test data of Con­
TraNet and TraNet. Each row of the figure shows the comparison of both
models for one subject. First column shows the training and validation
loss of ConTraNet, whereas second column shows the training and
validation loss of TraNet, whereas the third column in the figure shows
the comparison of validation loss of ConTraNet and TraNet. Lower
values of ConTraNet in comparison with TraNet indicate high general­
ization quality.
More concretely, Fig. 3 provides comprehensive comparison of both
the architecture. For all the four subjects, ConTraNet has shown the
same trend of moving close to its minimum while avoiding the over­
fitting, hence showing the ability to converge using only almost 400
samples for each subject. On the contrary, column two of Fig. 3 shows
the overfitting trend of TraNet for all the five subjects where its training
loss converges while the validation loss rises as the epochs increase. The
overfitting phenomenon is evident even though the number of training
parameters of TraNet are less compared to ConTraNet since it contains
only one encoder layer and uses the same regularization parameters as
ConTraNet. Column three of Fig. 3 shows the contrast between the
convergence of validation loss of ConTraNet and divergence of valida­
tion loss of TraNet as the epoch increases.
Fig. 3 shows the benefits of using a hybrid architecture like Con­
TraNet in low data setting for bio signals. It is because, contrary to
TraNet, ConTraNet did not need to learn all the inductive biases from
scratch since CNN block includes some of the inductive biases, which are
based on their design, in the architecture (see sections Introduction and
ConTraNet). This in turn aids the transformer block to focus more on
learning the context. However, in TraNet, there are no biases included in
its design. It requires TraNet to extract them using the limited available
training data in EEG and sEMG domain and consequently it suffers from
overfitting as well as divergence from the good local minimum.
3.2.1. Performance comparison with varying number of training examples
In addition, we performed a systematic analysis which decreases the
number of training examples iteratively and evaluate the respective
performance. For this analysis, in addition to ConTraNet, we considered
the following benchmark algorithms:
1. EEGNet [24].
2. ShallowNet [23].
The motivation behind the analysis is to study and analyze how
ConTraNet performs in low data settings. Meaning, what is the effect on
the decoding or classification performance of ConTraNet and its coun­
terparts when less amount of data is available for training and learning
the features from the signals. For the analysis, we used relatively large-
scale versatile dataset (Physionet MI-EEG) to evaluate the generalization
Fig. 2. Attention maps. a) Visualization of attention maps of the attention heads of transformer block showing the local and global context extracted by ConTraNet
from this input signal. x-axis shows the time samples of EEG signal and y-axis shows the expanded dimension of the attention maps for better visualization. Color
intensity in the plot shows the attention paid to the corresponding part of the signal. b) Mean attention map computed by calculating the mean of all attention heads
to visualize the mean attention paid on different parts of the signal.
O. Ali et al.


---

## Page 8

Computers in Biology and Medicine 168 (2024) 107649
8
Fig. 3. Training and Validation loss. Comparison of training process of ConTraNet and TraNet for 4 randomly selected subjects from dataset 2b, BCI competition
IV. Each row of the figure shows the training progress of ConTraNet and TraNet for one subject. X-axis of each plot shows the number of epochs used to train the
models. Y-axis shows the loss of each model.
O. Ali et al.


---

## Page 9

Computers in Biology and Medicine 168 (2024) 107649
9
quality of our algorithm and the above-mentioned benchmark algo­
rithms in low data settings. Since, Physionet MI-EEG dataset consists of
2, 3 and 4-class MI-tasks of 105 subjects. To ensure the transparency in
the reported result, we analyzed the performance of ConTraNet and
above-mentioned benchmark methods for each task separately.
For each MI-task (2, 3 or 4-class MI-task), the analysis is done in five
following steps using the data distribution shown in Fig. 4:
1. We begin the analysis by taking 50 % of the total data (data of ~52.5
subjects) for training and use remaining 50 % data (data of
remaining ~52.5 subjects) for evaluation.
2. In second step, we reduce the training data to 30 % (data of ~31.5
subjects) and use the remaining 70 % data (data of remaining ~73.5
subjects) for evaluation.
3. In the third step, the training data is further lowered to 20 % (data of
~21 subjects) and the remaining data of ~84 subjects is used for
evaluating the models.
4. In fourth step, 10 % data (data of ~10.5 subjects) is used for training
the models whereas the remaining 90 % data (data of ~94.5 sub­
jects) is used for the evaluation purpose.
5. In the final step, we took only 1 % data (data of ~1.04 subjects) for
training and used remaining 99 % data (data of ~104 subjects) for
evaluation.
Fig. 4. Data Configuration. Varying distribution of training and test data for the performance evaluation of ConTraNet, EEGNet and ShallowNet.
Fig. 5. Performance comparison of ConTraNet with other state of the art methods using different number of training samples on 2-class Physionet MI-EEG dataset on
cross subject decoding. First row of x-axis shows the percentage of total data used during training, whereas the 2nd row shows the exact number of training and
evaluation samples used.
O. Ali et al.


---

## Page 10

Computers in Biology and Medicine 168 (2024) 107649
10
More concretely, in case of 2-class MI-task (Left fist vs Right fist MI-
task), total amount of available data is 4410 trials. The data distribution
as shown in Fig. 4 for the task is as follows:
1. In first step, 2205 trials are used for training and remaining 2205
trials for evaluation.
2. In second step, 1323 trials are used for training and remaining 3087
trials for evaluation.
3. In third step, 882 trials are used for training and remaining 3528
trials for evaluation.
4. In fourth step, 441 trials are used for training and 3969 trials for
evaluation.
5. In last step of the analysis, only 44 trials are used for training and
remaining 4366 for evaluation.
Similarly, the analysis is performed for 3-class and 4-class MI-tasks as
well using the corresponding data distribution as per Fig. 4.
3.2.1.1. Cross-subject decoding performance on 2-class MI-task. It is pre­
sented in Fig. 5, that for different number of training trials, ConTraNet
outperformed its counterparts by achieving the top accuracy of 83.17 %
when trained on the data of 50 % subjects and tested on the remaining
50 % subjects. However, EEGNet and ShallowNet achieved 81.81 % and
78.73 % accuracy respectively. For the case, when only 10 % of the
subjects were used for training and the remaining 90 % were used for
evaluation, ConTraNet achieved 79.26 % accuracy which is 3.91 % less
than top accuracy of 83.17 %.
Whereas, for EEGNet and ShallowNet, the accuracy difference from
the top accuracy of 83.17 % is 5.09 % and 12.37 %. This result indicates
that generalization of the ConTraNet suffers least among its
counterparts.
Additionally, in the extreme scenario, where 1 % percent of entire
data (as shown by 0.01 entry of the bar plot in Fig. 5) results in the
training data of only one subject, results in the classification accuracy of
67.90 %, 63.70 %, and 56.94 % on test data (accumulated from 104
subjects) for ConTraNet, EEGNet and ShallowNet respectively. In this
case, the performance difference between ConTraNet and EEGNet is 4.2
% and the number is 10.96 % when compared to ShallowNet.
Based on the results showed in Fig. 5, we conclude that ConTraNet
achieves almost 80 % classification accuracy across subjects, when
trained on only 441 samples which results from 10 % percent of entire
data. Whereas EEGNet requires 1323 samples resulted from 30 %
percent of the entire data to cross the 80 % classification accuracy across
subjects. ShallowNet did not cross 80 % classification mark even on the
training data accumulated from half of the entire data (2205 training
samples). To achieve the 80 % evaluation accuracy across subjects,
EEGNet requires 3 times more training samples as compared to Con­
TraNet, whereas ShallowNet did not cross 80 % classification mark even
with 5 times more training samples than ConTraNet.
3.2.1.2. Cross-subject decoding performance on 3-class MI-tasks. Here, we
used the same protocol for distributing the data for training and eval­
uation as shown in Fig. 4. As shown in Fig. 6 ConTraNet achieves more
than 70 % cross-subject classification accuracy by using 30 % of the
entire dataset. Contrarily, EEGNet requires 50 % percent of the data to
achieve classification accuracy closer to 70 %. However, ShallowNet
could not achieve this classification benchmark even after using 50 %
data for training.
In the extreme scenario, where we used the data of only one subject
to train the models, ConTraNet achieved 49.29 % classification accuracy
on the remaining 104 subjects which is quite above the chance level for
3-class MI-tasks, whereas EEGNet achieved 46.91 % and the number is
40.43 % for ShallowNet. The classification difference between ConTra­
Net and EEGNet and ShallowNet in this case is 2.38 % and 8.86 %
Fig. 6. Performance comparison of ConTraNet with other state of the art methods using different number of training samples on 3-class Physionet MI-EEG dataset on
cross subject decoding. First row of x-axis shows the percentage of total data used during training, whereas the 2nd row shows the exact number of training and
evaluation samples used.
O. Ali et al.


---

## Page 11

Computers in Biology and Medicine 168 (2024) 107649
11
respectively.
3.2.1.3. Cross-subject decoding performance on 4-class MI-tasks. The
performance comparison of ConTraNet and its counterparts on 4-class
MI-tasks using different number of training samples is shown in Fig. 7.
In this case, ConTraNet requires 20 % (1764 samples) data of the entire
data for training to cross the 60 % classification accuracy across subjects.
Whereas EEGNet and ShallowNet did not cross 60 % even with 4410
training samples, equivalent to 50 % of the entire data. This data results
in 2.5 times more samples compared to what ConTraNet required.
Similarly, we also evaluated the performance of models using
training data of only one subject while testing them on the remaining
104 subjects. In this case, as Fig. 7 shows, ConTraNet acquired 37.25 %
classification accuracy which is 12.25 % above chance level. However,
EEGNet and ShallowNet achieved 35.31 % and 30.80 % accuracy which
is 10.31 % and 5.81 % above chance level in case of 4-class MI-tasks.
Based on the analyses presented in this section, we conclude that
ConTraNet learns to extract more generalized patterns and features from
the data which aids it to scale to large unseen dataset even after training
on limited trials. The performance difference between ConTraNet and its
counterparts becomes more evident as the complexity of the task in­
creases and the number of training trials decreases.
3.3. Classification of EEG and sEMG signals
Here, we show that, ConTraNet is not only able to extract local and
global context from the signals and converge using limited amount of
training data, but also able to perform better in the classification of
physiological signals compared to the conventional state-of-the-art
methods.
Four very popular and state-of-the-art methods, which follow the
end-to-end learning protocol, are chosen as baseline methods to
compare the performance with ConTraNet: EEGNet [24], DeepConvNet
[23], ShallowNet [23], 1D CNN [39]. In addition, we also built a CNN
and Long short-term memory (LSTM) based hybrid model called
CNN-LSTM and used it also as a baseline model since LSTM based ar­
chitectures are also used to extract context from time-series signals. A
brief explanation of CNN-LSTM architecture is presented in section
‘Appendix’ of the manuscript Additionally, we also compared the per­
formance of ConTraNet with deep architectures such as ResNet-50,
VGG-16, AlexNet, SqueezeNet and GoogLeNet as reported in Ref. [40].
Four versatile, publicly available benchmark datasets are used for
performance comparison and validation of the classification results:
• BCI Competition IV, dataset 2b
• Physionet MI-EEG dataset
• Mendeley Data – sEMG
• Mendeley Data-sEMG V1
The detailed description of the datasets is presented in subsection
‘Data description’ of section ‘Materials and Methods’.
3.3.1. Evaluation metrics and statistical analysis
We used accuracy as the evaluation metrics and performed paired t-
test to establish the statistical significance at significance levels 0.05 of
the proposed method compared to other state-of-the-art methods.
3.3.2. Evaluation protocol
Cross-subject generalization is a more challenging problem in EEG
and sEMG signals decoding compared to within subject decoding.
Fig. 7. Performance comparison of ConTraNet with other state of the art methods using different number of training samples on 4-class Physionet MI-EEG dataset on
cross subject decoding. First row of x-axis shows the percentage of total data used during training, whereas the 2nd row shows the exact number of training and
evaluation samples used.
O. Ali et al.


---

## Page 12

Computers in Biology and Medicine 168 (2024) 107649
12
Henceforth, the analyses in this section focus more on the performance
evaluation of ConTraNet and its counterparts in cross-subject decoding.
To have a fair comparison between state-of-the-art methods and the
proposed architecture, we performed K-fold cross validation on all
datasets. The evaluation protocol for datasets is as follows.
BCI Competition IV dataset 2b: Since the dataset comprised of MI-
EEG data of 9 with normal or corrected-to-normal vision subjects, here
we performed cross-subject decoding using 9-fold cross validation
(Global) resulting into 8 subjects for training and the remaining 1 sub­
ject for testing in each fold. Additionally, for this dataset we performed
with-in subject or subject specific decoding (SS) as well to analyze how
does the proposed architecture ConTraNet perform compared to its
counterparts on within subject decoding task. For other datasets, only
cross-subject decoding is performed. For this analysis, we trained Con­
TraNet on training sessions (Sessions 01T, 02T, and 03T) and evaluated
on test sessions (Sessions 04E and 05E) of each subject. Moreover, we
also evaluated the impact of transfer learning on the performance of
ConTraNet for each subject individually, where we finetuned the global
model of each fold obtained after cross validation, on the training ses­
sions of the respective left out subject (global to SS) and evaluated on the
test sessions.
Physionet MI-EEG dataset: The dataset comprised of MI-EEG data
of 109 healthy subjects. However, 4 subjects are excluded due to
imbalanced number of trials. Consequently, 105 subjects are used for
performance evaluation. Here, we performed cross-subject decoding
using 5-fold cross validation, resulting into 84 subjects for training and
the remaining 21 subjects for testing in each fold.
Mendeley Data – sEMG: The dataset includes EMG data of 40
healthy subjects. Here, we performed cross-subject decoding using 5-
fold cross validation which implies 32 subjects for training and the
remaining 8 subjects for testing in each fold.
Mendeley Data – sEMG V1: The dataset includes EMG data of 30
healthy subjects. Here, we performed cross-subject decoding using 5-
fold cross validation which implies 24 subjects for training and the
remaining 6 subjects for testing in each fold.
The summary of the datasets and the evaluation protocol used to
evaluate and compare the performance of ConTraNet with state-of-the-
art methods is presented in Table 2.
In the rest of the results section, we denote vanilla ConTraNet which
has 1 encoder layer in Transformer block and one convolutional layer
with 16 kernels in CNN block as ConTraNet1,16. Similarly, the vanilla
CNN-LSTM model which has only one convolutional layer with 32
kernels and one LSTM layer with 200 units is denoted by

## Cnn −Lstm1,32.

3.3.3. Performance evaluation on BCI competition IV dataset 2b
Here, we report the performance of ConTraNet1,16 on BCI Competi­
tion IV dataset 2b and its comparison with state-of-the-art methods.
Cross subject decoding: Table 3 shows the 9-fold cross validation
performance of each method. Table 3 shows that ConTraNet1,16 out­
performs all its counterparts. ConTraNet1,16 obtained 4.2 % improve­
ment in average classification performance compared to ShallowNet,
whereas the improvement is 2.9 % and 2.4 % compared to DeepConvNet
and EEGNet respectively. The numbers are 4.58 % and 16.31 %
compared to 1D CNN model and CNN −LSTM1,32 respectively.
The results also show that ConTraNet1,16 significantly outperformed
CNN −LSTM1,32 (p = 0.00006), ShallowNet (p = 0.007), DeepConvNet
(p = 0.025) and 1D CNN (p = 0.006), whereas its performance is not
significantly different from EEGNet (p = 0.055).
With-in subject decoding: Table 4 shows the with-in subject
decoding performance of ConTraNet1,16 and its comparison with state-
of-the-art methods. It also shows the effect of transfer learning on the
performance of ConTraNet1,16. It is seen from Table 4, that
ConTraNet1,16 obtained the highest average accuracy of 85.29 % which
is 27.37 % and 12.41 % higher than CNN −LSTM1,32 and 1D CNN
respectively. The difference is 5.57 %, 3.15 % and 2.38 % for Deep­
ConvNet, ShallowNet and EEGNet models respectively.
For with-in subject decoding, ConTraNet1,16 significantly out­
performed CNN −LSTM1,32 (p = 0.000025), ShallowNet (p = 0.041),
DeepConvNet (p = 0.017), 1D CNN (p = 0.0027) as well as EEGNet (p =
0.042).
Effect of transfer learning: Here, we studied the effect of transfer
learning on hybrid model ConTraNet and compared its performance
with other state-of-the-art methods. Since the effect of transfer learning
on deep CNN based architectures is well studied phenomena [41–44],
we considered only the two best performing CNN models EEGNet and
ShallowNet for the comparison with ConTraNet. In this analysis, we
finetuned the global model of each fold on the respective subject specific
data and the corresponding results are reported in Table 5.
Table 5 shows that finetuning of global models of each fold of
ConTraNet1,16 on respective subject specific data resulting into
ConTraNet1,16 -TL, yielded the average classification accuracy of 86.98
% which is 1.25 % and 1.5 % higher than average classification per­
formance of fine-tuned models of EEGNet -TL and ShallowNet - TL
respectively.
3.3.4. Performance evaluation on Physionet MI-EEG dataset
To further validate the performance of ConTraNet1,16, we evaluated
it’s results on another large scale publicly available dataset. Here, we
present the performance comparison of ConTraNet1,16 with state-of-the-
art methods on Physionet MI-EEG dataset. Since the dataset comprised
of left fist (L), right fist (R), both fists (B), both feet (F) MI-tasks as well as
the resting data (0), henceforth, the performance evaluation is made on
2-class (L/R), 3-class (L/R/0), and 4-class (L/R/0/F) MI-tasks.
Cross subject decoding: Table 6 shows the performance compari­
son of ConTraNet1,16 with state-of-the-art methods on 2-class, 3-class,
and 4-class MI-tasks on Physionet MI-EEG dataset. Table 6 shows that
ConTraNet1,16 outperforms its counterparts in classification of 2-class, 3-
class, and 4-class MI-tasks using 5-fold cross validation.
In 2-class MI-EEG decoding, ConTraNet1,16 obtained the average
classification accuracy of 83.61 % which is significantly higher (19.3 %)
than CNN −LSTM1,32. However, it is 3.52 % higher than 1D CNN and
3.09 % higher than ShallowNet, whereas it yielded an improvement of
2.67 % and 1.8 % compared to DeepConvNet and EEGNet respectively.
Here, ConTraNet1,16 significantly outperformed CNN −LSTM1,32 (p =
0.000045), ShallowNet (p = 0.0004), 1D CNN (p = 0.0062) and Deep­
ConvNet (p = 0.0428), whereas its performance is not significantly
different from EEGNet (p = 0.052).
In case of 3-class MI-EEG classification, ConTraNet1,16 achieved
74.38 % average decoding accuracy of 5-folds. Here, it showed an
Table 2
Summary of datasets and the evaluation protocol used in this study.
Dataset
Signal type
No. of Subjects (healthy)
No. of Classes
Evaluation Protocol
BCI Competition IV dataset 2b

## Mi-Eeg

9
2
Cross subject: 9-Fold CV,
Within subject (SS)
Physionet dataset

## Mi-Eeg

105
2, 3, 4
Cross subject: 5-Fold CV
Mendeley dataset
sEMG
40
10
Cross subject: 5-Fold CV,
Online decoding
Mendeley dataset V1
sEMG
30
7
Cross subject: 5-Fold CV
O. Ali et al.


---

## Page 13

Computers in Biology and Medicine 168 (2024) 107649
13
improvement of 27.77 %, 11.19 % and 2.97 % in average classification
accuracy compared to CNN −LSTM1,32, 1D CNN and ShallowNet
respectively. However, the improvement is 2.43 % and 1.93 %
compared to EEGNet and DeepConvNet respectively. For 3-class MI-EEG
decoding task, ConTraNet1,16 performed significantly better than
CNN −LSTM1,32 (p = 0.000058), ShallowNet (p = 0.0213), 1D CNN (p
= 0.0100) and EEGNet (p = 0.0341), however, it performed statistically
similar to DeepConvNet (p = 0.093).
In the event of classification of 4-class MI-EEG task, ConTraNet1,16
exhibited an average improvement of 13.12 % against the state-of-the-
art methods. It attained the average classification accuracy of 65.44 %
which implies an improvement of 30.94 % and 21.99 % compared to
CNN −LSTM1,32 and 1D CNN respectively whereas, 6.03 % compared to
ShallowNet. These numbers are 2.82 % and 3.80 % compared to
DeepConvNet and EEGNet respectively. In case of 4-class MI-EEG task,
our proposed architecture, significantly outperformed CNN −LSTM1,32
(p = 0.0000059), ShallowNet (p = 0.0004), 1D CNN (p = 0.0112),
DeepConvNet (p = 0.0432) as well as EEGNet (p = 0.0020).
3.3.5. Performance evaluation on Mendeley Data-sEMG
To further establish the significance of ConTraNet1,16 and its abilities
to learn distinct features across different HMI paradigms, we also eval­
uated its performance on a physiological dataset other than brain signals
(MI-EEG signals).
Cross Subject decoding: We opted for the data of muscle activity,
Mendeley Data-sEMG, for this purpose since it is a timeseries physio­
logical dataset. Here, we describe the performance comparison of state-
of-the-art methods with ConTraNet1,16 on the classification of 10-classes
of Mendeley Data-sEMG on cross-subjects using 5-fold cross validation.
The decoding performance of ConTraNet1,16 and the existing state-of-
the-art methods is presented in Table 7. It is shown in Table 7 that,
ConTraNet1,16 surpassed its counterparts by manifesting an average
improvement of 8.6 % on average classification accuracy of 10-class
EMG signals.
ConTraNet1,16 acquired an average classification accuracy of 77.15
% which showed an overwhelming higher classification difference of
53.6 % compared to CNN −LSTM1,32. With regards to the other models,
the performance of ConTraNet1,16 is 11.45 % higher than ShallowNet,
8.45 % greater than DeepConvNet and 6.10 % superior to EEGNet. In
case of sEMG dataset, ConTraNet1,16 significantly outperformed
CNN −LSTM1,32 (p = 0.00021), ShallowNet (p = 0.0052) and Deep­
ConvNet (p = 0.0211) and its performance is statistically similar to
EEGNet (p = 0.080).
Online decoding: In addition to using the complete trial of 6s for
cross-subject decoding, where 32 subjects were used for training and the
8 remaining subjects were used for testing in each fold, we also per­
formed online decoding. For online decoding, we followed the protocol
as presented by the authors of this dataset in Ref. [40]. In Ref. [40], the
authors of the sEMG Mendeley dataset removed the 1s from beginning
and 1s from the end of the sEMG signal to consider the maximal
contraction position as the main sEMG signal. The remaining 4s EMG
signals were segmented using sliding windows of 250 ms with the
window shift of 50 ms. Stratified k-fold cross validation (SKCV) is used
on the segmented EMG signals to evaluate the online-decoding perfor­
mance [40]. used Hilbert-Huang Transform (HHT) to obtain
time-frequency (TF) images from the segments and employed ResNet-50
(pre-trained on ImageNet dataset) model, which they fine-tuned on HHT
TF images. They also reported the performance comparison of other
deep neural networks. ResNet-50 is a deep architecture with over 23
million parameters, whereas the vanilla ConTraNet1,16 has only 267474
Table 3
Performance comparison of ConTraNet1,16 with state-of-the-art methods on Competition IV dataset 2b on cross-subject decoding.
Cross subject decoding: 9-Fold CV
2-class: L/R
Architecture
1
2
3
4
5
6
7
8
9
Avg
EEGNet
70.69
68.24
64.03
74.49
82.43
78.75
80.09
76.84
72.08
74.18
DeepConvNet
73.33
69.26
63.06
73.78
80.68
74.72
81.81
75.13
70.97
73.63
ShallowNet
70.56
66.32
62.78
75.95
76.89
71.25
80.03
73.82
73.06
72.29
1D CNN
68.06
66.79
64.17
80.00
76.49
68.89
78.61
77.76
67.22
71.99

## Cnn −Lstm1,32

56.39
54.71
52.50
67.16
61.62
56.81
60.00
70.00
63.19
60.26
ConTraNet1,16
72.92
72.94
63.75
83.51
82.70
80.69
84.44
77.37
70.83
76.57
Table 4
Performance comparison of ConTraNet1,16 with state-of-the-art methods on BCI Competition IV dataset 2b on with-in subject decoding.
With-in subject decoding (SS)
2-class: L/R
Architecture
1
2
3
4
5
6
7
8
9
Avg
EEGNet
68.12
68.44
85.94
97.81
95.31
72.19
89.06
94.38
75.00
82.91
DeepConvNet
55.00
67.50
83.13
94.38
86.56
71.25
87.50
92.19
80.00
79.78
ShallowNet
70.94
64.69
71.56
92.50
96.50
85.60
85.62
89.38
82.50
82.14
1D CNN
65.62
58.75
74.69
95.94
59.38
68.75
87.81
83.13
61.87
72.88

## Cnn −Lstm1,32

55.00
51.25
56.88
78.12
46.25
48.12
56.56
72.81
56.25
57.92
ConTraNet1,16
75.31
70.62
88.40
97.19
92.31
77.50
88.19
95.31
82.81
85.29
Table 5
Effect of transfer learning on ConTraNet and its comparison with other state-of-the-art models.
Transfer learning
2-class: L/R
Architecture
1
2
3
4
5
6
7
8
9
Avg
EEGNet - TL
70.94
71.88
81.88
97.19
97.19
81.56
94.38
95.63
80.94
85.73
ShallowNet -TL
74.37
67.19
85.62
93.44
96.25
84.06
90.62
93.44
84.38
85.48
ConTraNet1,16 -TL
76.88
72.15
89.00
98.00
95.00
87.50
90.30
95.00
79.00
86.98
O. Ali et al.


---

## Page 14

Computers in Biology and Medicine 168 (2024) 107649
14
parameters. To have a fair comparison with ResNet-50 and other deep
neural architectures, we also increased the depth of ConTraNet by
adding 7 encoder layers in transformer block and increased the con­
volutional filters from 16 to 32 in the convolutional layer of CNN block.
We did not increase the convolutional layers in CNN block. We denoted
the resulting model as ConTraNet7,32. Here 7 represents the encoder
layers in transformer block and 32 represents the convolutional filters in
CNN block. The number of parameters in ConTraNet7,32 are 498858
which are still significantly less than ResNet-50. We also increased the
LSTM layers in CNN-LSTM model to 2 to match the model parameters of
ConTraNet7,32. The resulting model is denoted as CNN −LSTM2,32 and
has 544342 parameters.
Table 8 shows the online-decoding performance comparison of
ConTraNet7,32
with
EEGNet,
DeepConvNet,
ShallowNet,
CNN −LSTM1,32, CNN −LSTM2,32, ResNet-50 and other deep architec­
tures reported in Ref. [40]. All the deep architectures reported in
Ref. [40] are pre-trained on large image datasets and fine-tuned on HHT
TF images of sEMG data, whereas ConTraNet7,32 and other remaining
architectures are trained from the scratch using the time series segments
of sEMG signals.
It is shown in Table 8, that ConTraNet7,32 outperformed ResNet-50
and all other deep architectures in online-decoding using 250 ms win­
dow segments of sEMG signals. The average validation accuracy ob­
tained by ConTraNet7,32 is 96.16 % while it achieved the average
training accuracy of 96.66 %, hence indicating its convergence to
minima without overfitting. The second-best performance is displayed
by ResNet-50 which scored 93.75 % average validation accuracy and is
2.41 % less than ConTraNet7,32. AlexNet, SqueezeNet and VGG-16
achieved above 80 % but below 90 % validation accuracy, whereas
GoogLeNet, VGG-10 and ResNet-34 scored above 90 % average training
as well as validation accuracy. In addition [40], also reported the
standard deviation of ResNet-50 on 5-fold SKCV which is 1.71 %,
whereas the standard deviation for ConTraNet7,32 is 0.23 % on valida­
tion accuracy, which is 1.48 % lower than ResNet-50. It also shows the
robustness of ConTraNet7,32 in comparison with ResNet-50 and other
deep architectures.
3.3.6. Performance evaluation on Mendeley Data-sEMG V1
To further validate the performance of ConTraNet in EMG domain,
we also evaluated its generalization results on another large scale pub­
licly available EMG dataset. Here, we present the performance com­
parison of ConTraNet1,16 with state-of-the-art methods on Mendeley
Data-sEMG V1.
Cross subject decoding: Here, we performed 5-fold cross validation
to evaluate the performance of ConTraNet and other state-of-the-art
methods for cross subject decoding. We employed two different set­
tings for this evaluation:
a. Setting 1: we used the complete 6s long trials for training as well as
evaluation.
b. Setting 2: we opted the pre-processing as suggested by the authors of
the dataset. In this case we removed 1s from the beginning and 1s
Table 6
Performance comparison of ConTraNet1,16 with state-of-the-art methods on 2-
class, 3-class, and 4-class Physionet MI-EEG dataset on cross-subject decoding
using 5-fold cross validation.
Cross subject decoding: 5-fold CV
2-class: L/R
Architecture
1
2
3
4
5
Avg
EEGNet
84.47
79.48
83.33
79.25
82.43
81.79
DeepConvNet
81.97
77.78
87.53
75.62
81.75
80.93
ShallowNet
80.73
78.12
83.67
77.44
82.65
80.52
1D CNN
82.20
78.23
81.29
78.00
80.73
80.09

## Cnn −Lstm1,32

62.02
62.81
66.89
64.51
65.31
64.31
ConTraNet1,16
84.24
81.18
87.07
80.61
84.61
83.61
Cross subject decoding: 5-fold CV
3-class: L/R/0
Architecture
1
2
3
4
5
Avg
EEGNet
74.07
70.60
75.74
69.92
69.39
71.94
DeepConvNet
70.52
71.96
77.17
71.05
71.50
72.44
ShallowNet
71.13
70.75
74.75
69.77
70.60
71.40
1D CNN
66.82
57.29
62.81
62.28
66.74
63.19

## Cnn −Lstm1,32

49.21
46.18
46.16
44.07
47.39
46.61
ConTraNet1,16
74.98
74.38
79.82
70.22
72.49
74.38
Cross subject decoding: 5-fold CV
4-class: L/R/0/F
Architecture
1
2
3
4
5
Avg
EEGNet
65.42
60.49
65.25
56.25
60.49
61.63
DeepConvNet
63.32
65.31
66.61
57.48
60.38
62.62
ShallowNet
59.86
58.73
62.19
56.86
59.41
59.41
1D CNN
52.72
32.77
33.11
47.39
51.25
43.45

## Cnn −Lstm1,32

34.92
34.92
35.94
33.16
33.56
34.50
ConTraNet1,16
67.23
64.51
69.27
61.05
65.14
65.44
Table 7
Performance comparison of ConTraNet with state-of-the-art methods on 10-class
Mendeley Data-sEMG on cross-subject decoding using 5-fold cross validation.
Cross subject decoding: 5-fold CV
10-class
Architecture
1
2
3
4
5
Avg
EEGNet
76.25
76.00
67.00
78.00
58.00
71.05
DeepConvNet
62.00
72.00
64.50
76.25
68.75
68.70
ShallowNet
59.00
69.75
64.50
69.25
66.00
65.70

## Cnn −Lstm1,32

29.25
26.00
27.00
10.00
25.50
23.55
ConTraNet1,16
78.50
79.25
74.50
79.75
73.75
77.15
Table 8
Online-decoding performance comparison of ConTraNet7,32 and other deep architectures as reported in Ref. [40] using stratified k-fold cross
validation.
Architecture
Feature
Train Acc (%)
Validation Acc (%)
SKCV 5-Fold
AlexNet

## Hht

82.31
81.56
SqueezeNet

## Hht

81.93
80.73
GoogLeNet

## Hht

91.34
90.36

## Vgg-16


## Hht

86.77
85.38

## Vgg-19


## Hht

92.56
90.65
ResNet-34

## Hht

96.05
92.23
ResNet-50

## Hht

96.12
93.75

## Cnn −Lstm1,32

Raw EMG
95.61
86.49

## Cnn −Lstm2,32

Raw EMG
85.72
80.79
EEGNet
Raw EMG
71.83
74.90
DeepConvNet
Raw EMG
75.11
41.69
ShallowNet
Raw EMG
72.48
65.95
ConTraNet7,32
Raw EMG
96.66
96.16
O. Ali et al.


---

## Page 15

Computers in Biology and Medicine 168 (2024) 107649
15
from the end of the sEMG signals to consider the maximum
contraction position of the muscles.
Table 9 shows the evaluation performance of ConTraNet and its
comparison with other state-of-the-art methods using setting 1. Here,
ConTraNet outperformed its counterparts with an average classification
accuracy of 85 % which is 2.57 % higher than EEGNet (p = 0.2608) and
5.14 % higher than ShallowNet (p = 0.1651). However, it yielded a
significant improvement of 9.36 % and 45.07 % compared to Deep­
ConvNet (p =
0.0411) and CNN −LSTM1,32 (p =
0.000140)
respectively.
Table 10 presents the results of evaluation performance of ConTra­
Net and its counterparts using setting 2. Here ConTraNet significantly
outperformed all the other state-of-the-art models by achieving an
average classification accuracy of 83.64 %. It is significantly higher than
CNN −LSTM1,32 (39.95 %) (p = 0.0071) and DeepConvNet (16.85 %)
(p = 0.0123). However, it achieved an improvement of 4.78 % and 9.78
% compared to EEGNet (p = 0.0349) and ShallowNet (p = 0.0089)
respectively.
The results presented in Table 3, Table 4, Table 6, Table 7, Table 8,
Table 9, and Table 10 show that ConTraNet generalizes well on cross-
subject as well as within-subject analyses for different physiological
signals compared to other state-of-the-art methods. They also show that
ConTraNet can extract and learn distinct features from different HMI
domains. ConTraNet not only showed better performance compared to
architectures which primarily extract local dependencies such as EEG­
Net, DeepConvNet, ShallowNet and 1D CNN, but it also showed ad­
vantages over CNN-LSTM network which also extracts context or long-
range dependencies from the time-series signals. ConTraNet and CNN-
LSTM both use a single convolutional layer which extracts the higher-
level features from the input signal and the long-range context must
be learned from these high-level features. The results indicate that,
ConTraNet can robustly extract global context even from the high-level
features compared to CNN-LSTM which struggles with its performance.
Moreover, CNN-LSTM architecture performed well in online decoding of
sEMG signals when the input signal is 250 ms long as shown in Table 8.
This also proves the theoretical notion that LSTM based architectures
work well with shorter sequences compared to longer sequences due to
its forgetfulness and vanishing gradient problem. On the contrary,
ConTraNet showed consistent performance both with short and long
input sequences.
3.4. Significance of CNN and transformer blocks in ConTraNet
In this experiment, we evaluated the significance of CNN and
Transformer blocks in ConTraNet separately. To do so, we trained and
evaluated two separate models namely ConNet and TraNet. ConNet
contains only the CNN block without Transformer block and TraNet
contains only the Transformer block without CNN block. ConNet and
TraNet are trained similarly to ConTraNet. The evaluation performance
of both the models is shown in Table 11.
Table 11 indicates the impact of removing the CNN and Transformer
blocks from ConTraNet on the average classification performance. When
CNN block is removed (TraNet), the average classification accuracy
drops from 85.29 % to 76.46 % resulting in 8.83 % drop in average
performance. Whereas, when the Transformer block is removed (Con­
Net), the average classification accuracy drops to 78.78 % which is
equivalent to 6.51 % drop in average performance. The reason for this
accuracy drop is twofold: firstly, removing the CNN block removes the
inductive biases from the overall architecture, which is now learned by
the TraNet from scratch. The amount of training data required to learn
the inductive biases from scratch is quite less. Secondly, the local and
deep features that are learned by the CNN block are now absent from the
TraNet which also must be learned from scratch using the limited
amount of data. Similarly, when the Transformer block is removed
(ConNet), the ability of the model to look for global dependencies is
restricted by the kernel length in CNN block which leads the model to
learn only local features and consequently drops the average perfor­
mance. The performance of TraNet is 2.32 % lower than ConNet. The
weaker performance of TraNet compared to ConNet is expected because
of the limited amount of training data. TraNet is a purely Transformer
based architecture which generally requires more data to learn inductive
biases, local and global features from scratch [30].
3.5. Frequency representation of the filtered signals from the learned
kernels
Here, we scrutinized the question, what did the CNN block of Con­
TraNet learn from the raw EEG signals. The CNN block performs
convolution operations on the raw signals by convolving fixed length
kernels across the signal, and thus outputs the filtered signals. Since the
filters are learned during training through back propagation, it learns to
filter the signals into significant frequencies which play the vital role in
successful classification of the signals. Henceforth, it can be implied that
the CNN block learns to extract the meaningful frequency components
which results into correct mapping of raw signal into the user intent.
This operation in CNN block is analogous to the pre-processing step
in conventional BCI pipelines, where band-pass filtering is applied to
filter the raw time-series signals into user defined frequency bands. It is
Table 11
Significance of CNN and transformer block in ConTraNet.
Sub
ConNet

## (Ss)

TraNet

## (Ss)

ConTraNet

## (Ss)

1
67.50
67.81
75.31
Subject specific decoding 2-
class L/R
2
66.87
70.31
70.62
3
84.06
85.94
88.40
4
93.44
81.88
97.19
5
86.56
87.50
92.31
6
71.56
70.00
77.50
7
83.75
87.19
88.19
8
88.75
84.38
95.31
9
66.56
53.12
82.81
Avg
78.78
76.46
85.29
Table 10
Performance comparison of ConTraNet with state-of-the-art methods on 7-class
and 4s long trials of Mendeley Data-sEMG V1 on cross-subject decoding using 5-
fold cross validation.
Cross subject decoding: 5-fold CV
7-class; 4s signals
Architecture
1
2
3
4
5
Avg
EEGNet
82.49
80.35
78.57
82.49
70.36
78.86
DeepConvNet
71.07
67.85
71.42
75.35
48.21
66.79
ShallowNet
76.07
81.43
72.14
74.64
65.00
73.86

## Cnn −Lstm1,32

68.21
38.21
51.79
46.07
14.2
43.69
ConTraNet1,16
88.21
85.71
77.50
88.57
78.21
83.64
Table 9
Performance comparison of ConTraNet with state-of-the-art methods on 7-class
and 6s long trials of Mendeley Data-sEMG V1 on cross-subject decoding using 5-
fold cross validation.
Cross subject decoding: 5-fold CV
7-class; 6s signals
Architecture
1
2
3
4
5
Avg
EEGNet
88.90
83.57
81.78
82.85
75.00
82.43
DeepConvNet
71.42
74.28
73.92
79.28
79.28
75.64
ShallowNet
84.28
82.49
78.21
87.85
66.42
79.86

## Cnn −Lstm1,32

40.00
54.64
37.50
31.78
35.71
39.93
ConTraNet1,16
84.64
89.29
88.57
83.92
78.57
85.00
O. Ali et al.


---

## Page 16

Computers in Biology and Medicine 168 (2024) 107649
16
Fig. 8. Learned frequency components. a) Mean filtered signal for each channel learned by the CNN block from the entire dataset. x-axis represents time samples
and y-axis represents the amplitude information b) The frequency components for each channel extracted by the CNN from the corresponding mean filtered signal.
Here x-axis represents the frequency components from 0 Hz till Nyquist frequency and y-axis represents the amplitude.
O. Ali et al.


---

## Page 17

Computers in Biology and Medicine 168 (2024) 107649
17
also similar to an operation where, Short-time Fourier Transform (STFT)
is applied to the raw signal to extract the existing frequency components
from it and later choose the frequency components that are relevant to
correctly classify the signals. However, in case of CNN block, the rele­
vant frequencies or bands are learned during training instead of defining
a priori as in case of conventional BCI pipelines.
To interpret the average frequency components learned by the CNN
block, ConTraNet which is trained on the entire dataset is employed to
obtain the feature maps of the CNN block over the whole dataset. The
mean feature map is then computed which represents the mean filtered
signal of the dataset which is shown in Fig. 8 (a). Fast Fourier Transform
(FFT) is then applied on mean filtered signal of each channel to obtain
the average frequency components learned by the CNN block over the
entire dataset as shown in Fig. 8 (b).
It is shown in Fig. 8, that the average frequency components learned
by the CNN block of ConTraNet are below 30 Hz which coincides with
the theoretical notion that motor imagery (MI) information resides sub
30Hz [45–47]. However, it also extracted the frequency components
from theta band which ranges between 4 and 8 Hz which also coincides
with the frequency bands extracted in Refs. [22,38], and [40].
4. Discussion and Summary
This study proposes a hybrid deep learning algorithm to decode MI-
EEG and sEMG signals. The proposed algorithm conjugate CNN and
transformer to extract meaningful patterns and a fully connected layer to
map the resultant features space to decision space.
In contrast to conventional MI-EEG and sEMG decoding frameworks
[45,48,49], but like many others state of the art frameworks [23,24,27],
the proposed algorithm is an end-to-end machine learning model. In
conventional decoding frameworks, the process of mapping raw signals
to features space and then mapping of resultant features space to deci­
sion spaces are exclusive. Filter Bank Common Spatial Pattern (FBCSP),
Short Time Fourier Transform (STFT), anchored-STFT [22] and wavelet
transform are among the most valuable feature extraction methods for
MI-EEG and sEMG signals [22,45,46,49]. Similarly, k-Nearest Neighbor
(kNN), Naïve Bayesian Panzer Window, Fisher Linear Discriminant,
Support Vector Machine (SVM) are among the most used machine
learning models for inference [45,49].
Even though, the conventional methods perform reasonably well in
translating MI-EEG and sEMG signals, yet they pose a potent drawback
in their methodology, which is the hand engineered features. To alle­
viate this problem, end-to-end deep Learning (DL) algorithms like CNN
is one possible replacement. CNN based algorithms like EEGNet [24],
DeepConvNet, ShallowNet [23], ResNet and VGG [40] provide
state-of-the-art results on benchmark publicly available datasets (BCI
Competition IV dataset 2b, Physionet MI-EEG dataset, Mendeley
Data-sEMG, and Mendeley Data-sEMG V1).
Shared kernels in CNN extract the features which possess temporal
invariance, which could be critical for non-stationary signals like EEG
and sEMG. However, the fixed size kernels restrict the learning of long-
range global dependencies as well as deep features simultaneously.
Moreover, the weight sharing in CNN architectures attend to the entire
signal uniformly. It limits the implication of stand-alone CNN based
architectures in extracting long-range and short-range dependencies,
simultaneously, which are critical in decoding neurophysiological sig­
nals like EEG and sEMG.
To overcome this limitation of CNN, we cascaded the transformer
block on top of CNN block to encapsulate time invariance, short and
long-range dependencies simultaneously. Furthermore, inclusion of
transformer block highlights the crucial part of the signal whereas
suppress the less relevant part of the signals due to the attention
mechanism integrated into the transformer block. Since the inductive
bias induced by the CNN block in the ConTraNet ensures the require­
ment of less training data to learn long-range dependencies and to
perform attention by the transformer block. This hybrid schema not only
addresses the limitations of the CNN architectures but also the short
comings of the standalone transformer architectures, such as requiring
large corpus of data to train.
The systematic quantitative analysis presented in section 3.2 mani­
fests the significance of our hybrid architecture ConTraNet in low data
settings. The results indicate that ConTraNet performs significantly
better than its stand-alone CNN counterparts in varied complex tasks (2-
class, 3- class and 4-class). Even in extreme case, where only 1 % data
(data of only one subject) is used for training in 2-class, 3-class, and 4-
class tasks, ConTraNet achieved 67.90 %, 49.29 % and 37.25 % evalu­
ation accuracies respectively. Whereas these numbers are 63.70 %,
46.91 %, 35.31 % for EEGNet and 56.94 %, 40.43 %, 30.89 % for
ShallowNet. We believe that improvement in ConTraNet is due to the
following reason: the CNN block uses the small and fixed receptive field
to extract deep and local features, whereas the transformer block de­
termines the interconnections and dependencies between those local
features, which, in turn provides the long-range dependencies that are
crucial for successful decoding of EEG and sEMG signals. Absence of any
of these components would adversely affect the performance of the ar­
chitecture in low data settings. Increasing the depth of stand-alone CNN
based architectures (ResNet and VGG) such that early layers use small
receptive field while later layers use large receptive fields can mitigate
to some extent, the issue of learning local and large range dependencies
simultaneously in large data settings. However, in presence of less
training data, the issue of overfitting is prevalent among deep CNN ar­
chitectures. Therefore, it is highly unlikely to employ stand-alone deep
CNN architectures when the available data is scarce.
Similarly, using stand-alone transformer architecture with less
training data would result in poor performance as presented in Table 11.
We believe that the poor performance of stand-alone transformer ar­
chitecture (TraNet) is due to the following reasons: the stand-alone
transformer architecture lacks the ability to prioritize any hypothesis
in the hypotheses space by its design. It must learn all the generalized
rules in the hypotheses space which are essential for the successful
decoding of EEG and sEMG signals from scratch. Since stand-alone
transformer architecture consists of attention-mechanism as a sole
extractor of the rules from the hypotheses space, it therefore requires
seeing large amount of data during training to infer generalized rules for
the correct mapping and decoding of the unseen data. Henceforth it is
also not recommended to use stand-alone transformer architecture in
low data settings.
The significance of the hybrid architecture ConTraNet is also evident
from the decoding performance on varied complex tasks and on two
different physiological signals as reported in sections 3.3.3, 3.3.4,
3.3.5, and 3.3.6. The results indicate that ConTraNet performed
significantly better than its counterparts and the performance difference
grew as the complexity of the task enhanced. Here increased complexity
refers to increase in number of classes to distinguish. The results indicate
that, by increasing the number of classes, the evaluation performance of
ConTraNet is less affected compared to other state-of-the-art methods.
We believe that it is due to the ability of the architecture to learn to
extract the interconnections between different parts of the signal. It
therefore highlights the vital components and parts across entire signal
which are crucial for its correct decoding. It also suppresses the less
O. Ali et al.


---

## Page 18

Computers in Biology and Medicine 168 (2024) 107649
18
relevant parts of the signal. It can be considered as assigning a signifi­
cance factor to each part of the signal (like assigning a value to each
word in speech to determine the main context). Consequently, it aids the
last mapping layer to correctly map the signal from feature space to the
decision space. However, the counterparts of ConTraNet lack this ability
and look uniformly at the entire signal to extract features for decoding
which in turn can confound the model and result in poor performance.
Another important question is that how good transformer are in
comparison with other models that can learn temporal dependencies
like LSTMs. To answer this question, performance comparison is made
between ConTraNet and another hybrid model CNN-LSTM. This com­
parison indicates that ConTraNet outperformed its counter hybrid model
by large margin as shown in Table 3 to Table 8. Even though, CNN-LSTM
follows the hybrid schema, yet it performs poorly compared to Con­
TraNet. The reason we believe is that LSTM based architectures are
effective for short sequences, whereas for long sequences these models
incur forgetfulness in the architecture due to vanishing gradient and
long-connectivity problem between current state and the early states
which results in their poor performances on long sequences [50]. Con­
nectivity in network is defined as the length of the paths, forward and
backward, a signal has to travel in the network. It is a vital factor
influencing the ability to learn long-range dependencies. The shorter
these paths are, the easier it is to learn long-range dependencies. Since
LSTM networks are recurrent in nature, the connectivity paths are
longer which affects their ability to learn long-range dependencies in
long sequences and hence resulting in poor performance [30]. On the
contrary, transformer architecture allows parallelism in the network
which solves the problem of long connectivity. It uses a self-attention
mechanism to learn the global dependencies in a sequence effectively
without compromising on computational and statistical efficiency.
Transformer architecture works equally well on short as well as long
sequences.
This phenomenon of CNN-LSTM is also evident from Tables 7 and 8.
In the analysis presented in Table 7, entire EMG signals are employed at
once which are 6 s long each. Using such long signals with CNN-LSTM,
affected its decoding performance tremendously since LSTM block could
not retain all the information due to vanishing gradient problem.
However, in the analysis shown in Table 8, each EMG signals is
segmented into chunks of 250 ms which are then used for training and
evaluating the models. In this case, CNN-LSTM performed better
compared to some other state-of-the-art architectures. We believe that
the reason is the short sequence of the signals in this analysis. It helped
CNN-LSTM to retain the information using short-connectivity and
perform better compared to few other methods. On the contrary, Con­
TraNet performed better with both long as well as short sequences.
Above explained results summarize that the ConTraNet is robust to
extract and learn the distinct and more generalized features from
different neurophysiological signals (EEG and EMG) for both long as
well as short sequences.
Another important aspect of ConTraNet is its requirement of less
training data to handle multiple modalities of signals simultaneously for
neurorehabilitation applications. Here, we considered two most used
modalities (EEG and sEMG) of non-invasive nature. This property of
ConTraNet makes it a strong candidate to be deployed in real-time
neurorehabilitation scenarios where multi modalities are deployed for
neurorehabilitation and data scarcity is prevailing. It is also concluded
that in scenarios, where large amount of data is available, and where
memory constraints are not a prevailing issue, deeper versions of Con­
TraNet can be used to attain better performances.
In addition, ConTraNet builds upon CNN and transformer networks
and same end-to-end pipeline is used for both neurophysiological
domains, which makes it ideal candidate to be used for wearable edge
devices such as neuromorphic chips which have memory as well as
computational constraints. In addition, the in-built parallelism of the
transformer block can positively impact the usability of ConTraNet
framework on edge devices.
Limitations: However, this study has few limitations as well. One
limitation of this work is that ConTraNet is trained separately for EEG
and EMG signals, which can enhance the total computational and
memory requirements. However, in future work, we plan to work on a
generalized network by jointly training the model on EEG and EMG
signals, which results into more and versatile training data and less
computational resource requirement. Training on large and versatile
data could increase the model robustness against different neurophysi­
ological domains. Another limitation of the ConTraNet is that it is
constrained by the memory limitations of the modern hardware and
wearable edge devices such as neuromorphic chips. Increasing the
length of the input sequence or signal, would result into greater number
of patches which are extracted after CNN block. Since all the patches
(which are also called as tokens in NLP domain) are used in parallel by
the self-attention layer in transformer encoder block, therefore its
memory requirements are quadratic in the number of patches (as the
computational complexity of self-attention layer is of the order O(n2.d);
n = number of patches and d = dmodel). It implies that doubling the
number of patches would quadruple the computational and memory
requirements. However, in future we plan to work on optimizing the
attention mechanism by selective parallelism which can result into less
computational resources.
5. Conclusion
Neurophysiological signals, particularly EEG and EMG, manifest
time-varying and non-stationary features. They exhibit short as well as
long-range dependencies and are predominately used in neuro­
rehabilitation process. In addition, subjective variability in the data
further complicates the decoding process. In our previous work [22], we
attempted to extract robust features and aimed to create a machine
learning model able to deal with such complexities using limited data.
Despite our efforts in our previous study and several other studies
[23,24,40], learning the local as well as global dependencies from the
data remained evasive due to the inherent structure of extracted features
and CNN model. To address this challenge, our current study focuses on
employing an end-to-end learning pipeline utilizing hybrid design based
on CNN and transformer architectures. This novel approach nourishes
the strength of CNN and transformers, simultaneously. CNN provides the
features which includes the inductive bias and as a result with limited
training data transformer successfully learns the local and global de­
pendencies. It therefore achieves a robust solution for the decoding of
EEG and EMG signals.
Additionally, our current approach devolves on CNN and trans­
former networks, making it particularly suitable for edge devices like
neuromorphic chips. Looking ahead, embedding our hybrid solution on
neuromorphic chips holds the potential to significantly benefit the
multi-modal HMI paradigm in rehabilitating severely movement
impaired patients.
Declaration of competing interest
The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.
O. Ali et al.


---

## Page 19

Computers in Biology and Medicine 168 (2024) 107649
19
Acknowledgement
This work is supported by the Ministry of Economics, Innovation,
Digitization and Energy of the State of North Rhine-Westphalia and the
European Union, grants GE-2-2-023A (REXO) andIT-2-2-023 (VAFES).
Appendix
Convolutional Neural Network (CNN)
A CNN consists of convolutional layers as its backbone. The layer parameters focus on the use of learnable kernels which are essential to auto­
matically extract the meaningful features from raw inputs. The process of feature extraction takes place in several stages by stacking multiple con­
volutional layers. A convolutional layer first learns to map the raw input into low-level features, which are then mapped to mid-level features. Mid-
level features are then mapped to high-level features. High-level representation of the input suppresses the redundant information and highlights the
distinguishable pivotal information.
In addition to the convolutional layers as the main building block of a CNN architecture, it may also contain few other layers [51] such as
normalization layers which ensure that the gradients while training remain in a certain range, activation layers which add the non-linearity to the
output of a convolutional layer, pooling layer which reduces the dimension of the feature maps and extracts the most relevant features from the output
of convolutional layer, regularization layer to avoid overfitting and fully connected layer which is analogous to standard artificial neural networks
(ANNs) in its role.
Similarly, ConTraNet also consists of a CNN block with convolutional layer, normalization layer, activation layer, and pooling layer.
CNNs have produced many state-of-the-art results in solving computer vision tasks such as image classification, object detection and image
segmentation etc. One reason behind this success is the translation invariance or equivariance of CNNs. Secondly, the ability of the CNNs to learn deep
features by using small size kernels also play a vital role in many patterns’ recognition tasks. However, at the same time, the small size kernels limit its
ability to learn long rage dependencies which is usually vital in time series signals such as MI-EEG and sEMG signals. This issue is generally addressed
by increasing the receptive field of the architecture by stacking more convolutional layers, hence deepening the model. Increasing the depth of the
model also enhances the trainable parameters which in result increases the computational cost and the chances of overfitting. Overfitting can badly
hamper the generalization ability of the learned model and consequently its overall performance.
Batch normalization in CNN
Why is batch normalization required: During the training of neural networks, the distribution of the input to each layer changes due to the
change in the parameters of the previous layer. This phenomenon of change in the distribution of inputs to layers of network during the training
process is referred as Internal Covariate Shift [52,53].
The Covariate Shift presents a problem as the layers must continuously adapt to the new distribution which requires lower learning rates and
careful parameter initialization. This as a result slows down the training process tremendously and makes it hard to train the models. Therefore,
minimizing the internal covariate shift as much as possible ensures the faster training and introduces stability in the training process. By minimizing
the internal covariate shift, we mean minimizing the change in the distribution of layer’s input.
Henceforth, to deal with the internal covariate shift, Batch Normalization (BN) is introduced in Ref. [52].
How batch normalization works in CNN: BN helps to stabilize the model during training and ensures that the network training converges faster.
It achieves it by applying normalization to fix the mean and the variance of the inputs to the layer which results in fixed distribution of layer’s input.
BN is used as a neural network layer. It takes input from layer below and applies whitening (normalizing to have zero mean and unit variance) to
the inputs and forward them as input to layer above. This process helps to achieve the fixed distribution of the layer’s input. It therefore tries to
minimize the internal covariate shift.
BN has following parameters.
• Learnable parameters β and γ.
• Non-learnable parameters μ (batch mean) and σ (batch variance).
During training process, at each step, one mini batch of the data is processed by the network. BN layer processes the data as follows.
1. BN gets the activations (feature maps) of all the samples in a mini batch from the previous layer as input.
2. Firstly, for all the activations (feature maps) in mini batch corresponding to the same convolutional kernel, it computes the μi and σi jointly across
the mini batch.
3. Then for each activation (feature map), it normalizes the data using the corresponding μi and σi values. It results into zero mean and unit variance.
4. Then the normalized values are scaled and shifted by element-wise multiplication with the γi and addition with βi.
Transformers
Transformers are originally introduced for machine translation in Ref. [30] and have since become the de facto standard method for many NLP
asks. The network architecture mainly relies on an attention mechanism and disregards the recurrence and convolutions completely. The sole reliance
on attention gives it the ability to draw global dependencies between the input and the output. The standard transformer architecture as shown in
Fig. 9 consists of an encoder and a decoder block. Transformer-encoder module encodes the input, which is 1D sequence of token embeddings, to a
sequence of continuous representations. Whereas the Transformer-decoder generates the output sequence one element at a time provided the encoded
representation from encoder.
O. Ali et al.


---

## Page 20

Computers in Biology and Medicine 168 (2024) 107649
20
Fig. 9. Transformer model architecture. This figure is modified after [30].
Encoder block: Transformer-encoder consists of several identical layers called encoder-layers. Each encoder-layer has two sub-layers namely
multi-head self-attention layer and position wise fully connected feed-forward layer. Each sub-layer is followed by layer normalization [54]. The input
to each sublayer is connected to the followed normalization layer via residual connection [55].
Decoder block: Like encoder, decoder also consists of several identical layers called decoder-layers. Each decoder layer has three sub-layers. Two
of them are multi-head self-attention layers and the remaining is the fully connected feed-forward layer. One of the multi-head attention layers attends
to the encoded representation of the encoder block, whereas the other multi-head attention layer has masking mechanism which prevents the decoder
positions to attend to subsequent positions which ensures that the output for position i can only depend on the outputs of positions less than position i.
Similar to encoder block, each sub-layer is followed by layer normalization and the connection between the input to each sub-layer and the following
normalization layer is established through the residual connection.
Self-attention: The attention as shown in Fig. 10 maps the query and a set of key-value pairs to an output. The query, key and value vectors are
obtained through the linear projection of input vector using three linear layers. The attention as shown in equation (1) is then performed by
computing the dot products of query with all the keys, which are then scaled by the dimension of the model which is same as the dimension of query
and keys. The softmax function is then applied to the result to obtain the weights on values. The resultant is finally multiplied by the value vector to
obtain the complete attention score.
Attention(Q, K, V) = softmax

## (Qkt

̅̅̅̅
dk
√
)

## V

(1)
O. Ali et al.


---

## Page 21

Computers in Biology and Medicine 168 (2024) 107649
21
Fig. 10. Multi-Head attention. Schematic diagram of Scaled Dot-Product Attention (left) and Multi-Head Attention (right). This figure is modified after [30].
Multi-Head attention: Multi-head attention layer as shown in Fig. 10 consists of multiple self-attention mechanisms that run in parallel and are
referred as heads. Here the queries, keys and values are linearly projected h times with different linear projections. Self-attention function is then
performed on each of the projected versions of queries, keys, and values in parallel which are concatenated and once again projected to output the
result. Multi-Head attention can be represented as equation (2).
Multi_Head(Q, K, V) = Concatenate(head1, …, headh)WO
(2)
where headi = Attention(Q WQ
i , K WK
i ,VWV
i )
Here WQ
i ∈Rdmodel×dk, WK
i ∈Rdmodel×dk, WV
i ∈Rdmodel×dv and WO ∈Rhdv×dmodel are the projection matrices. dmodel is the dimension of the model and h
represents number of heads.
Embeddings and Positional Encoding: Embeddings are the learned representations of input and output tokens in form of a vector that meets the
dimension criterion of the model. Transformer model is permutation invariant because it does not contain any recurrence and convolution operations.
To add some information about the position of the tokens in the sequence, positional encodings are added to the input embeddings in both encoder and
decoder blocks.

## Cnn-Lstm

CNN-LSTM is a hybrid model consisting of convolutional layer and LSTM layers. It has one convolutional layer with 32 kernels of size (1,125). The
stride of 1 is used with ‘valid’ padding. ReLU activation is used in convolutional layer. Additionally, it has one LSTM layer with 200 units and one
dense layer with 100 units and an output layer corresponding to the number of outputs.
The CNN-LSTM network used in this study is inspired from other similar architectures in this research [56,57]. Since the architectures presented in
these studies are not publicly available, we implemented in this study a similar CNN-LSTM network based on their architectural designs for the
analyses mentioned above.
Hyperparameters tuning during training of ConTraNet
The ConTraNet model explained in section 2.2.1 ConTraNet is a deep learning model. It requires several hyperparameters which are tuned using
grid search method. We employed BCI competition IV dataset 2b for the tuning of the hyperparameters. Since it comes with separate training and
evaluation session, we used training sessions to tune the hyperparameters of the ConTraNet. The results corresponding to some of the hyperparameters
selection process are presented here.
Selection of convolutional kernels in CNN block: Table 12 shows the impact of different number of convolutional kernels in CNN block of
ConTraNet. It is shown in Table 12, ConTraNet achieved 85.29 % average classification accuracy using 16 convolutional kernels which is 2.73 % and
2.48 % better compared to using 8 and 32 kernels respectively. Since 16 kernels achieved best performance, we opted this as a hyperparameter value
for ConTraNet.
Table 12
Selection of number of convolutional kernels of CNN block in ConTraNet.
Number of kernels
1
2
3
4
5
6
7
8
9
Avg
8
71.55
65.94
83.75
96.87
89.05
78.75
85.00
93.75
78.43
82.56
16
75.31
70.62
88.40
97.19
92.31
77.50
88.19
95.31
82.81
85.29
32
75.62
68.12
87.18
96.88
90.93
73.43
83.44
93.75
75.94
82.81
O. Ali et al.


---

## Page 22

Computers in Biology and Medicine 168 (2024) 107649
22
From Table 12, we infer that the low performance of ConTraNet using 8 and 32 convolutional kernels compared to 16 kernels is ought to be due to
underfitting and overfitting respectively. With 8 convolutional kernels, the trainable parameters of CNN block are close to 1000, which can incur
underfitting in the model whereas, the trainable parameters of CNN block are quadrupled (4000) using 32 convolutional kernels resulting into
overfitting in the CNN block which effects the overall performance of the ConTraNet. The results indicate that 16 convolutional kernels produce the
optimal results with 2000 trainable parameters in CNN block.
Selection of patch width in transformer block: In addition, we evaluated the impact of patch width in the transformer block. The impact of
patches of different size in terms of evaluation accuracy is shown in Fig. 11 (a, b). Patches of different sizes in transformers ensure the inclusion of local
context in the extracted information. However, bigger patch size and sliding window with overlapping stride can also add redundancy. As a result, the
generalization quality can be compromised, which can be seen in Fig. 11, in the case of larger patch sizes (3,4,5). Here, evaluation accuracy is reduced
by almost 2 % as compared to patch of smaller size (2). However, using patch width of 1, fails adding local context in the input to the transformer
encoder. As a result, the learning of local dependencies is affected, which is also shown in Fig. 11. Therefore, we used the smaller patch size (2) with
the stride 1 as it yielded the best overall accuracy.
Fig. 11. Impact of patch width in transformer block on the performance of ConTraNet.
O. Ali et al.


---

## Page 23

Computers in Biology and Medicine 168 (2024) 107649
23
References
[1] G. Andreoni, S. Parini, L. Maggi, L. Piccini, G. Panfili, A. Torricelli, Human machine
interface for healthcare and rehabilitation, in: S. Vaidya, L.C. Jain, H. Yoshida
(Eds.), Advanced Computational Intelligence Paradigms in Healthcare-2, vol. 65,
Springer Berlin Heidelberg, Berlin, Heidelberg, 2007, pp. 131–150, https://doi.
org/10.1007/978-3-540-72375-2_7. Studies in Computational Intelligence, vol. 65.
[2] A. Kübler, A. Furdea, S. Halder, E.M. Hammer, F. Nijboer, B. Kotchoubey, A brain-
computer interface controlled auditory event-related potential (P300) spelling
system for locked-in patients, Ann. N. Y. Acad. Sci. 1157 (1) (Mar. 2009) 90–100,
https://doi.org/10.1111/j.1749-6632.2008.04122.x.
[3] A.B. Ajiboye, et al., Restoration of reaching and grasping movements through
brain-controlled muscle stimulation in a person with tetraplegia: a proof-of-
concept demonstration, Lancet 389 (10081) (May 2017) 1821–1830, https://doi.
org/10.1016/S0140-6736(17)30601-3.
[4] C. Castellini, A.E. Fiorilla, G. Sandini, Multi-subject/daily-life activity EMG-based
control of mechanical hands, J. NeuroEng. Rehabil. 6 (1) (Dec. 2009) 41, https://
doi.org/10.1186/1743-0003-6-41.
[5] D. Xiong, D. Zhang, X. Zhao, Y. Zhao, Deep learning for EMG-based human-
machine interaction: a review, IEEECAA J. Autom. Sin. 8 (3) (Mar. 2021) 512–533,
https://doi.org/10.1109/JAS.2021.1003865.
[6] B. Graimann, B. Allison, G. Pfurtscheller, Brain–computer interfaces: a gentle
introduction, in: B. Graimann, G. Pfurtscheller, B. Allison (Eds.), Brain-Computer
Interfaces, The Frontiers Collection., Springer Berlin Heidelberg, Berlin,
Heidelberg, 2009, pp. 1–27, https://doi.org/10.1007/978-3-642-02091-9_1.
[7] A. Bottomley, Myo-electric control of powered prostheses, J. Bone Joint Surg. Br.
47 (3) (1965) 411–415.
[8] L.F. Nicolas-Alonso, J. Gomez-Gil, Brain computer interfaces, a review, Sensors 12
(2) (Jan. 2012) 1211–1279, https://doi.org/10.3390/s120201211.
[9] S. Kellis, K. Miller, K. Thomson, R. Brown, P. House, B. Greger, Decoding spoken
words using local field potentials recorded from the cortical surface, J. Neural. Eng.
7 (5) (Oct. 2010), 056007, https://doi.org/10.1088/1741-2560/7/5/056007.
[10] T. Aflalo, et al., Decoding motor imagery from the posterior parietal cortex of a
tetraplegic human, Science 348 (6237) (May 2015) 906–910, https://doi.org/
10.1126/science.aaa5417.
[11] J.J. Daly, J.R. Wolpaw, Brain–computer interfaces in neurological rehabilitation,
Lancet Neurol. 7 (11) (Nov. 2008) 1032–1043, https://doi.org/10.1016/S1474-
4422(08)70223-0.
[12] E.W. Sellers, D.B. Ryan, C.K. Hauser, Noninvasive brain-computer interface enables
communication after brainstem stroke, Sci. Transl. Med. 6 (257) (Oct. 2014),
https://doi.org/10.1126/scitranslmed.3007801.
[13] R. Merletti, D. Farina, Analysis of intramuscular electromyogram signals, Philos.
Trans. R. Soc. Math. Phys. Eng. Sci. 367 (1887) (Jan. 2009) 357–368, https://doi.
org/10.1098/rsta.2008.0235.
[14] M. Chen, P. Zhou, A novel framework based on FastICA for high density surface
EMG decomposition, IEEE Trans. Neural Syst. Rehabil. Eng. 24 (1) (Jan. 2016)
117–127, https://doi.org/10.1109/TNSRE.2015.2412038.
[15] F. Cincotti, et al., Non-invasive brain-computer interface system: towards its
application as assistive technology, Brain Res. Bull. 75 (6) (Apr. 2008) 796–803,
https://doi.org/10.1016/j.brainresbull.2008.01.007.
[16] C. Neuper, G.R. Müller-Putz, R. Scherer, G. Pfurtscheller, Motor imagery and EEG-
based control of spelling devices and neuroprostheses, Prog. Brain Res. 159 (2006)
393–409, https://doi.org/10.1016/S0079-6123(06)59025-9.
[17] G. Pfurtscheller, G.R. Müller, J. Pfurtscheller, H.J. Gerner, R. Rupp, ’Thought’–
control of functional electrical stimulation to restore hand grasp in a patient with
tetraplegia, Neurosci. Lett. 351 (1) (Nov. 2003) 33–36, https://doi.org/10.1016/
s0304-3940(03)00947-9.
[18] S.M. Rissanen, M. Koivu, P. Hartikainen, E. Pekkonen, Ambulatory surface
electromyography with accelerometry for evaluating daily motor fluctuations in
Parkinson’s disease, Clin. Neurophysiol. Off. J. Int. Fed. Clin. Neurophysiol. 132
(2) (Feb. 2021) 469–479, https://doi.org/10.1016/j.clinph.2020.11.039.
[19] J. Moron, T. DiProva, J.R. Cochrane, I.S. Ahn, Y. Lu, EMG-based hand gesture
control system for robotics, in: 2018 IEEE 61st International Midwest Symposium
on Circuits and Systems (MWSCAS), IEEE, Windsor, ON, Canada, Aug. 2018,
pp. 664–667, https://doi.org/10.1109/MWSCAS.2018.8624056.
[20] M. Saif-ur-Rehman, et al., SpikeDeeptector: a deep-learning based method for
detection of neural spiking activity, J. Neural. Eng. 16 (5) (Oct. 2019), 056003,
https://doi.org/10.1088/1741-2552/ab1e63.
[21] M. Saif-ur-Rehman, et al., SpikeDeep-Classifier: a deep-learning based fully
automatic offline spike sorting algorithm, J. Neural. Eng. (Nov. 2020), https://doi.
org/10.1088/1741-2552/abc8d4.
[22] O. Ali, M. Saif-ur-Rehman, S. Dyck, T. Glasmachers, I. Iossifidis, C. Klaes,
Enhancing the decoding accuracy of EEG signals by the introduction of anchored-
STFT and adversarial data augmentation method, Sci. Rep. 12 (1) (Dec. 2022)
4245, https://doi.org/10.1038/s41598-022-07992-w.
[23] R.T. Schirrmeister, et al., Deep learning with convolutional neural networks for
EEG decoding and visualization: convolutional Neural Networks in EEG Analysis,
Hum. Brain Mapp. 38 (11) (Nov. 2017) 5391–5420, https://doi.org/10.1002/
hbm.23730.
[24] V.J. Lawhern, A.J. Solon, N.R. Waytowich, S.M. Gordon, C.P. Hung, B.J. Lance,
EEGNet: a compact convolutional neural network for EEG-based brain–computer
interfaces, J. Neural. Eng. 15 (5) (Oct. 2018), 056013, https://doi.org/10.1088/
1741-2552/aace8c.
[25] Y. LeCun, Y. Bengio, Convolutional networks for images, speech, and time series,
The handbook of brain theory and neural networks 3361 (10) (1995).
[26] Y. Song, X. Jia, L. Yang, L. Xie, Transformer-based Spatial-Temporal Feature
Learning for EEG Decoding (2021), https://doi.org/10.48550/ARXIV.2106.11170.
[27] G. Dai, J. Zhou, J. Huang, N. Wang, HS-CNN: a CNN with hybrid convolution scale
for EEG motor imagery classification, J. Neural. Eng. 17 (1) (Jan. 2020), 016025,
https://doi.org/10.1088/1741-2552/ab405f.
[28] S. An, S. Kim, P. Chikontwe, S.H. Park, Few-shot relation learning with attention
for EEG-based motor imagery classification, in: 2020 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS), Las Vegas, IEEE, NV, USA, Oct.
2020, pp. 10933–10938, https://doi.org/10.1109/IROS45743.2020.9340933.
[29] Y. Ding, N. Robinson, S. Zhang, Q. Zeng, C. Guan, TSception: capturing temporal
dynamics and spatial asymmetry from EEG for emotion recognition, IEEE Trans.
Affect. Comput. (2022), https://doi.org/10.1109/TAFFC.2022.3169001, 1–1.
[30] A. Vaswani, et al., Attention Is All You Need, 2017, https://doi.org/10.48550/

## Arxiv.1706.03762.

[31] J. He, L. Zhao, H. Yang, M. Zhang, W. Li, Hsi-Bert, Hyperspectral image
classification using the bidirectional encoder representation from transformers,”,
IEEE Trans. Geosci. Rem. Sens. 58 (1) (Jan. 2020) 165–178, https://doi.org/
10.1109/TGRS.2019.2934760.
[32] A. Dosovitskiy, et al., An Image Is Worth 16x16 Words: Transformers for Image
Recognition at Scale, 2020, https://doi.org/10.48550/ARXIV.2010.11929.
[33] A.L. Goldberger, , et al.PhysioToolkit PhysioBank, PhysioNet, Components of a
new research resource for complex physiologic signals,”, Circulation 101 (23) (Jun.
2000) https://doi.org/10.1161/01.CIR.101.23.e215.
[34] M.A. Ozdemir, D.H. Kisa, O. Guren, A. Akan, Dataset for multi-channel surface
electromyography (sEMG) signals of hand gestures, Data Brief 41 (Apr. 2022),
107921, https://doi.org/10.1016/j.dib.2022.107921.
[35] Mehmet Akif Ozdemir, Dataset for multi-channel surface electromyography
(sEMG) signals of hand gestures, Mendeley, Sep. 15 (2021), https://doi.org/
10.17632/CKWC76XR2Z.1.
[36] M. Tangermann, et al., Review of the BCI competition IV, Front. Neurosci. 6
(2012), https://doi.org/10.3389/fnins.2012.00055.
[37] D.-A. Clevert, T. Unterthiner, S. Hochreiter, Fast and Accurate Deep Network
Learning by Exponential Linear Units, ELUs), 2015, https://doi.org/10.48550/

## Arxiv.1511.07289.

[38] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, Dropout: a
simple way to prevent neural networks from overfitting, J. Mach. Learn. Res. 15
(56) (2014) 1929–1958.
[39] F. Mattioli, C. Porcaro, G. Baldassarre, A 1D CNN for high accuracy classification
and transfer learning in motor imagery EEG-based brain-computer interface,
J. Neural. Eng. 18 (6) (Dec. 2021), 066053, https://doi.org/10.1088/1741-2552/
ac4430.
[40] M.A. Ozdemir, D.H. Kisa, O. Guren, A. Akan, Hand gesture classification using
time–frequency images and transfer learning based on CNN, Biomed. Signal
Process Control 77 (Aug. 2022), 103787, https://doi.org/10.1016/j.
bspc.2022.103787.
[41] J. M. Williams, “Deep Learning and Transfer Learning in the Classification of EEG
Signals” .
[42] N. Ho, Y.-C. Kim, Evaluation of transfer learning in deep convolutional neural
network models for cardiac short axis slice classification, Sci. Rep. 11 (1) (Jan.
2021) 1839, https://doi.org/10.1038/s41598-021-81525-9.
[43] A. Kensert, P.J. Harrison, O. Spjuth, Transfer learning with deep convolutional
neural networks for classifying cellular morphological changes, SLAS Discov 24 (4)
(Apr. 2019) 466–475, https://doi.org/10.1177/2472555218818756.
[44] Y. Zheng, J. Huang, T. Chen, Y. Ou, W. Zhou, Transfer of learning in the
convolutional neural networks on classifying geometric shapes based on local or
global invariants, Front. Comput. Neurosci. 15 (Feb. 2021), 637144, https://doi.
org/10.3389/fncom.2021.637144.
[45] Kai Keng Ang, Zhang Yang Chin, Haihong Zhang, Cuntai Guan, Filter Bank
common spatial pattern (FBCSP) in brain-computer interface, in: 2008 IEEE
International Joint Conference on Neural Networks (IEEE World Congress on
Computational Intelligence), IEEE, Hong Kong, China, Jun. 2008, pp. 2390–2397,
https://doi.org/10.1109/IJCNN.2008.4634130.
[46] Y.R. Tabar, U. Halici, A novel deep learning approach for classification of EEG
motor imagery signals, J. Neural. Eng. 14 (1) (Feb. 2017), 016003, https://doi.org/
10.1088/1741-2560/14/1/016003.
[47] F. Li, F. He, F. Wang, D. Zhang, Y. Xia, X. Li, A novel simplified convolutional
neural network classification algorithm of motor imagery EEG signals based on
deep learning, Appl. Sci. 10 (5) (Feb. 2020) 1605, https://doi.org/10.3390/
app10051605.
[48] F. Lotte, M. Congedo, A. L´ecuyer, F. Lamarche, B. Arnaldi, A review of
classification algorithms for EEG-based brain–computer interfaces, J. Neural. Eng.
4 (2) (Jun. 2007) R1–R13, https://doi.org/10.1088/1741-2560/4/2/R01.
[49] S.K. Bashar, M.I.H. Bhuiyan, Classification of motor imagery movements using
multivariate empirical mode decomposition and short time Fourier transform
based hybrid method, Eng. Sci. Technol. Int. J. 19 (3) (Sep. 2016) 1457–1464,
https://doi.org/10.1016/j.jestch.2016.04.009.
[50] S. Li, W. Li, C. Cook, C. Zhu, Y. Gao, Independently recurrent neural network
(IndRNN): building A longer and deeper rnn, in: 2018 IEEE/CVF Conference on
Computer Vision and Pattern Recognition, IEEE, Salt Lake City, UT, Jun. 2018,
pp. 5457–5466, https://doi.org/10.1109/CVPR.2018.00572.
[51] K. O’Shea, R. Nash, An Introduction to Convolutional Neural Networks (2015),
https://doi.org/10.48550/ARXIV.1511.08458.
[52] S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift, 2015, https://doi.org/10.48550/

## Arxiv.1502.03167.

O. Ali et al.


---

## Page 24

Computers in Biology and Medicine 168 (2024) 107649
24
[53] H. Shimodaira, Improving predictive inference under covariate shift by weighting
the log-likelihood function, J. Stat. Plann. Inference 90 (2) (Oct. 2000) 227–244,
https://doi.org/10.1016/S0378-3758(00)00115-4.
[54] J.L. Ba, J.R. Kiros, G.E. Hinton, Layer Normalization (2016), https://doi.org/
10.48550/ARXIV.1607.06450.
[55] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in:

### 2016 IEEE Conference On Computer Vision and Pattern Recognition (CVPR), Las

Vegas, IEEE, NV, USA, Jun. 2016, pp. 770–778, https://doi.org/10.1109/

## Cvpr.2016.90.

[56] F.M. Garcia-Moreno, M. Bermudez-Edo, M.J. Rodriguez-Fortiz, J.L. Garrido,
A CNN-LSTM deep learning classifier for motor imagery EEG detection using a low-
invasive and low-cost BCI headband, in: 2020 16th International Conference on
Intelligent Environments (IE), IEEE, Madrid, Spain, Jul. 2020, pp. 84–91, https://
doi.org/10.1109/IE49459.2020.9155016.
[57] P. Lu, N. Gao, Z. Lu, J. Yang, O. Bai, Q. Li, Combined CNN and LSTM for motor
imagery classification, in: 2019 12th International Congress on Image and Signal
Processing, BioMedical Engineering and Informatics (CISP-BMEI), IEEE, Suzhou,
China, Oct. 2019, pp. 1–6, https://doi.org/10.1109/CISP-

## Bmei48845.2019.8965653.

O. Ali et al.