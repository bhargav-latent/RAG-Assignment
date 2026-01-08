# vector symbolic finate state machines in Attractor Neural Networks

*Extracted using NVIDIA Nemotron-Parse*

---

## Page 1

# Vector Symbolic Finite State Machines in Attractor Neural Networks

1,2,3 , High Greatorex 2,3 , Martin Ziegler 1, and Elisabetta Chicca 2,3

### Abstract

Hopfield attractor networks are robust distributed models of human memory, but lack a general mechanism for effecting state-dependent attractor transitions in response to input. We propose construction rules such that an attractor network may implement an arbitrary finite state machine (FSM), where states and stimuli are represented by high-dimensional random vectors, and all state transitions are enacted by the attractor network’s dynamics. Numerical simulations show the capacity of the model, in terms of the maximum size of implementable FSM, to be linear in the size of the attractor network for dense bipolar state vectors, and approximately quadratic for sparse binary state vectors. We show that the model is robust to imprecise and noisy weights, and so a prime candidate for implementation with high-density but unreliable devices. By endowing attractor networks with the ability to emulate arbitrary FSMs, we propose a plausible path by which FSMs could exist as a distributed computational primitive in biological neural networks.

**Keywords:** Attractor network, vector symbolic architectures, hyperdimensional computing, finite state machine, neural state machine

### I. INTRODUCTION

Hopfield attractor networks are one of the most celebrated models of robust neural auto-associative memory, as from a simple Hebbian learning rule they display emergent attractor dynamics which allow for reliable pattern recall, completion, and correction even in situations with considerable non-idealities imposed (Amit 1989; Hopfield 1982). Attractor models have since found widespread use in neuroscience as a functional and tractable model of human memory (Chaudhuri

and Fiete 2016; Eliasmith 2005; Khona and Fiete 2022; Little 1974; Rolls 2013; Scneidman et al. 2006). The assumption of these models is that the network represents different states by different, usually uncorrelated, global patterns of persistent activity. When the network is presented with an input that closely resembles one of the stored states, the network state converges to the corresponding fixed-point attractor.

This process of switching between discrete attractor states is thought to be fundamental both to describe biological neural activity, as well as to model higher cognitive decision making processes (Brinkman et al. 2022; Daelli and Treves 2010; Mante et al. 2013; Miller 2016; Tajima et al. 2017). What attractor models currently lack, however, is the ability to perform state-dependent computation, a hallmark of human cognition (Buonomano and Maass 2009; Dayan 2008; Granger 2020). That is, when the network is presented with an input, the attractor state to which the network switches ought to be dependent both upon the input stimulus as well as the state the network currently inhabits, rather than simply the input.

We thus seek to endow a classical neural attractor model, the Hopfield network, with the ability to perform state-dependent switching between attractor states, without resorting to the use of biologically implausible mechanisms, such as training via backpropagation algorithms. The resulting attractor networks will then be able to robustly emulate any arbitrary Finite State Machine (FSM), considerably improving their usefulness as a neural computational primitive.

We achieve this by leaning heavily on the framework of Vector Symbolic Architectures (VSAs), also known as Hyper- dimensional Computing (HDC). VSAs treat computation in an entirely distributed manner, by letting symbols be represented by high-dimensional random vectors, hypervectors (Gayler 1998; Kanerva 1997; Kleyko, Rachkovskij, et al. 2022; Plate 1995). When equipped with a few basic operators for binding and superimposing hypervectors together, corresponding often either to component-wise multiplication or addition respectively, these architectures are able to store primitives such a sets, sequences, graphs and arbitrary data bindings, as well as enabling more complex relations, such as analogical and figurative reasoning (Kanerva 2009; Kleyko, Davies, et al. 2021). Although different VSA models often have differing representations and binding operations (Kleyko, Rachkovskij,

1

<sup>1</sup>Micro- and Nanoelectronic Systems, Institute of Micro- and Nanotechnologies (IMN) MacroNano®, Technische Universität Hmenau, Ilmenau, Germany. <sup>2</sup>Bio-Inspired Circuits and Systems (BICS) Lab, Zernike Institute for Advanced Materials, University of Groningen, Netherlands. <sup>3</sup>Groningen Cognitive Systems and Materials Center (CogniGron), University of Groningen, Netherlands.

Madison Cotteret

arXiv:2212.01196v2 [cs.NE] 14 Dec 2023

This is the authors’ final version before publication in Neural Computation. We thank Dr. Federico Corradi, Dr. Nicoletta Risi and Dr. Matthew Cook for their invaluable input and suggestions, as well as their help with proofreading this document. Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project MemTDE Project number 441959088 as part of the DFG priority program SPP 2262 MemrisTec - Project number 422738993, and Project NMVAC - Project number 432009531. The authors would like to acknowledge the financial support of the CogniGron research center and the Ubo Emmius Funds (Univ. of Groningen).

---

## Page 2

<tbc>et al. 2022), they all share the need for an auto-associative cleanup memory, which can recover a clean version of the most similar stored hypervector, given a noisy version of itself. We here use the recurrent dynamics of a Hopfield-like attractor neural network as a state-holding auto-associative memory (Gritsenko et al. 2017).

Symbolic FSM states will thus be represented each by a hypervector and stored within the attractor network as a fixed-point attractor. Stimuli will also be represented by hypervectors, which, when input to the attractor network, will trigger the network dynamics to transition between the correct attractor states. We make use of common VSA techniques to construct a weights matrix to achieve these dynamics, where we use the Hadamard product between bipolar hypervectors {-1,1}_<sup>N</sup>_ as the binding operation (the Multiply-Add-Permute (MAP) VSA model) (Gayler 1998). We thus claim that attractor-based FSMs are a plausible biological computational primitive insofar as Hopfield networks are.

This represents a computational paradigm that is a departure from conventional von Neumann architectures, wherein the separation of memory and computation is a major limiting factor in current advances in conventional computational performance (the von Neumann bottleneck (Backus 1978; Indiveri and Liu 2015)). Similarly, the high redundancy and lack of reliance on individual components makes this architecture fit for implementation with novel in-memory computing technologies such as resistive RAM (RRAM) or phase-change memory (PCM) devices, which could perform the network’s matrix-vector‐multiplication (MVM) step in a single operation (Ielmini and Wong 2018; Xia and Yang 2019; Zidan and Lu 2020).

## II. METHODS

### A. Hypervector arithmetic

Throughout this article, symbols will be represented by high-dimensional randomly-generated dense bipolar hypervectors<tbc>

\(\mathbf{x}\in\{-1,1\}^N\) (1)

<tbc>where the number of dimensions _N_ is generally taken to be greater than 10,000. Unless explicitly stated otherwise, any bold lowercase Latin letter may be assumed to be a new, independently generated hypervector, with the value _Y<sub>i</sub>_ at any index _i_ in **x** generated according to<tbc>

\(\text{IP}(Y_i=1)=\text{IP}(Y_i=-1)=\frac{1}{2}\) (2)

<tbc>For any two arbitrary hypervectors **a** and **b**, we define the similarity between the two hypervectors by the normalised inner product

\(d(\mathbf{a},\mathbf{b}):=\frac{1}{N}\mathbf{a}^\unknown\mathbf{b}=\frac{1}{N}\sum_{i=1}^Na_ib_i\) (3)

where the similarity between a hypervector and itself _d_(**a**,**a**)=1, and _d_(**a**,-**a**)=-1. Due to the high dimensionality of the hypervectors, the similarity between any two<tbc>

<tbc>unrelated (and so independently generated) hypervectors is the mean of an unbiased random sequence of -1 and 1s<tbc>

\(d(\mathbf{a},\mathbf{b})=0\pm\frac{1}{\sqrt{N}}\approx 0\) (4)

<tbc>which tends to 0 for \(N\rightarrow\infty\). It is from this result that we get the requirement of high dimensionality, as it ensures that the inner product between two random hypervectors is approximately 0. We can thus say that independently generated hypervectors are _pseudo-orthogonal_ (Kleyko, Davies, et al. 2021). For a set of independently generated states \(\{\mathbf{x}^\mu\}\), these results can be summarised by<tbc>

\(d(\mathbf{x}^\mu,\pm\mathbf{x}^\nu)=^{N\rightarrow\infty}\pm\delta^{\mu\nu}\) (5)

<tbc>where \(\delta^{\mu\nu}\) is the Kronecker delta. Hypervectors may be combined via a so called _binding_ operation to produce a new hypervector that is dissimilar to both its constituents. We here choose the Hadamard product, or component-wise multiplication, as our binding operation, denoted "\(\circ\)". Article number, page 45 of 58

\((\mathbf{a}\circ\mathbf{b})_i=a_i\cdot b_i\) (6)

The statement that the binding of two hypervectors is dissimilar to its constituents is written as

\(d(\mathbf{a}\circ\mathbf{b},\mathbf{a})\approx 0d(\mathbf{a}\circ\mathbf{b},\mathbf{b})\approx 0\) (7)

where we implicitly assume that _N_ is large enough that we can ignore the \(\mathcal{O}(\frac{1}{\sqrt{N}})\) noise terms. If we wish to recover a similarity between the hypervectors \(\mathbf{a}\circ\mathbf{b}\) and **a**, we could bind the **b** hypervector to the lone **a** term as well, in which case we would have \(d(\mathbf{a}\circ\mathbf{b},\mathbf{a}\circ\mathbf{b})=1\). For reasons of ease and robustness of implementation in an asynchronous neural system, we focus instead on another method to recover the similarity (see Sections III-A & VII-C). If we _mask_ the system using **b**, such that only components where _b<sub>i</sub>_=1 are remaining. Then, we have

\(d\big(\mathbf{a}\circ\mathbf{b},\mathbf{a}\circ H(\mathbf{b})\big)=\frac{1}{N}\sum_{1\le i\le N}a_ib_ia_iH(b_i)=\frac{1}{N}\big[\sum_{1\le i\le N\overline{b_i}=1}a_i^2H(1)-\sum_{1\le i\le N\overline{b_i}=-1}a_i^2H(-1)\big]=\frac{1}{N}\sum_{1\le i\le Ny\overline{b_i}=1}1\approx\frac{1}{2}\text{(8)}\)

where we have used the Heaviside step function \(H(\cdot)\) defined by

\(\big(H(\mathbf{b})\big)_i=H(b_i)=\left\{\begin{array}{ c c c }
1 & \text{if} & b_i>0 \\
0 & \text{otherwise}
\end{array}\right.\) (9)

to create a multiplicative mask _H_(**b**), setting to 0 all components where _b<sub>i</sub>_=-1. In the second line, we have split the summation over all components into summations over<tbc>

2

---

## Page 3

<tbc>components where _b<sub>i</sub>_=1 and -1 respectively. The final similarity of \(\frac{1}{2}\) is a consequence of approximately half of all values in a any hypervector being +1 (Equation 2).

### B. Hopfield networks

A Hopfield network is a dynamical system defined by its internal state vector **z** and fixed recurrent weights matrix **W**, with a state update rule given by<tbc>

\(\mathbf{z}_{t+1}=\mathrm{sgn}\big(\mathbf{W}\mathbf{z}_t\big)\) (10)

<tbc>where **z**_<sub>t</sub>_ is the network state at discrete time step _t_, and \(\mathrm{sgn}(\cdot)\) is an component-wise sign function, with zeroes resolving<sup>1</sup> to +1 (Hopfield 1982). We know that if we want to store _P_ uncorrelated patterns \(\{\mathbf{x}^\nu\}_{\nu=1}^P\) within a Hopfield network, we can construct the weights matrix **W** according to

\(\mathbf{W}=\sum_{\mathrm{patterns}\nu}^P\mathbf{x}^\nu\mathbf{x}^{\nu\unknown}\) (11)

then as long as not too many patterns are stored (_P\<_0.14_N_ (Hopfield 1982)), the patterns will become fixed-point attractors of the network’s dynamics, and the network can perform robust auto-associative pattern completion and correction.

### C. Finite State Machines

A Finite State Machine (FSM) _M_ is a discrete system with a finite state set \(X_\mathrm{FSM}=\{\chi_1,\chi_2,\dots,\chi_{N_Z}\}\), a finite input stimulus set \(S_\mathrm{FSM}=\{\varsigma_1,\varsigma_2,\dots,\varsigma_{N_S}\}\), and finite output response set \(R_\mathrm{FSM}=\{\rho_1,\rho_2,\dots,\rho_{N_R}\}\). The FSM _M_ is then fully defined with the addition of the transition function \(F(\cdot):X_\mathrm{FSM}\times S_\mathrm{FSM}\rightarrow X_\mathrm{FSM}\) and the output response function \(G(\cdot):X_\mathrm{FSM}\times S_\mathrm{FSM}\rightarrow R_\mathrm{FSM}\)

\(\begin{array}{ c c }
x_{t+1}=F(x_t,s_t) &  \\
r_{t+1}=G(x_t,s_t) & 
\end{array}\) (12)

where \(x_t\in X_\mathrm{FSM}\), \(r_t\in R_\mathrm{FSM}\) and \(s_t\in S_\mathrm{FSM}\) are the state, output and stimulus at time step _t_ respectively. The transition function \(F(\cdot)\) thus provides the next state for any state-stimulus pair, while \(G(\cdot)\) provides the output, and both may be chosen arbitrarily. The FSM _M_ can thus be represented by a directed graph, where each node represents a different state \(\chi\), and every edge has a stimulus \(\varsigma\) and optional output \(\rho\) associated with it.

## III. ATTRACTOR NETWORK CONSTRUCTION

We now show how a Hopfield-like attractor network may be constructed to emulate an arbitrary FSM, where the states within the FSM are stored as attractors in the network, and the stimuli for transitions between FSM states trigger all corresponding transitions between attractors. More specifically, for every FSM state \(\chi\in X_\mathrm{FSM}\), an associated hypervector **x** is randomly generated and stored as an attractor within the network, the set of which we denote _X_<sub>AN</sub>. We henceforth refer to these hypervectors as node hypervectors, or node<tbc>

<tbc>attractors. Every unique stimulus \(\varsigma\in S_\mathrm{FSM}\) in the FSM is also now associated with a randomly generated hypervector \(\mathbf{s}\in S_\mathrm{AN}\), where _S_<sub>AN</sub> is the set of all stimulus hypervectors. For the FSM edge outputs \(\rho\in R_\mathrm{FSM}\), a corresponding set of output hypervectors \(\mathbf{r}\in R_\mathrm{AN}\) is similarly generated. These correspondences are summarised in  I.

### A. Constructing transitions

We consider the general situation that we want to initiate a transition from _source_ attractor state \(\mathbf{x}\in X_\mathrm{AN}\) to _target_ attractor state \(\mathbf{y}\in X_\mathrm{AN}\), by imposing some stimulus hypervector \(\mathbf{s}\in S_\mathrm{AN}\) as input onto the network.

\(\mathbf{x}\longrightarrow^\mathbf{s}\mathbf{y}\) (13)

To ensure the plausible functionality of the network in a biological system, the mechanism for enacting transitions in the network should make very few timing assumptions about the system, and should be robust to an arbitrary degree of asynchrony. How we model input to the network is thus of crucial importance to its functionality in these regimes. We model input to the network as a _masking_ of the network state, such that all components where the stimulus **s** is -1 are set to 0. This may be likened to saying we are considering input to the network that selectively silences half of all neurons according to the stimulus hypervector. This mechanism was chosen as it allows the network to function even when the input is applied asynchronously and with random delays (see Section VII-C). While a stimulus hypervector **s** is being imposed upon the network, the modified state update rule is given by

\(\mathbf{z}_{t+1}=\mathrm{sgn}\big(\mathbf{W}(\mathbf{z}_t\circ H(\mathbf{s}))\big)\) (14)

where the Hadamard product of the network state with _H_(**s**) enacts the masking operation, and the weights matrix **W** is constructed such that **z**<sub>_t_+1</sub> will resemble the desired target state (Section III-A).

For every edge in the FSM, we randomly generate an "edge state" **e**, which is also stored as an attractor within the network. Each edge will use this **e** state as an intermediate attractor state, en route to **y**. Additionally, each unique stimulus \(\varsigma\in S_\mathrm{FSM}\) will now have _two_ stimulus hypervectors associated with it, **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_, which trigger transitions from source state **x** to edge state **e** and edge state **e** to target state **y** respectively. The edge states are introduced to allow the system to function even when stimuli are input to the network for arbitrarily many time steps, and prevents unwanted effects such as skipping<tbc>

<sup>1</sup>Though this arbitrary choice may seem to incur a bias to a particular state, in practise the postsynaptic sum very rarely equals 0.

3

\begin{tabular}{ccc}
\multicolumn{2}{c}{FSM (Symbols)} & Attractor Network (Hypervectors) \\
States & \(\chi\in X_\mathrm{FSM}\) & Attractors \(\mathbf{x}\in X_\mathrm{AN}\) \\
Stimuli & \(\varsigma\in S_\mathrm{FSM}\) & Stimuli \(\mathbf{s}\in S_\mathrm{AN}\) \\
Outputs & \(\rho\in R_\mathrm{FSM}\) & Outputs \(\mathbf{r}\in R_\mathrm{AN}\) \\
\end{tabular}

TABLE I: A comparison of the notation used to represent states, stimuli and outputs in the FSM, and the corresponding hypervectors used to represent the FSM within the attractor network.

---

## Page 4

over certain attractor states, or oscillations between states (see Section VII-D). A general transition now looks like

\(\mathbf{x}\longrightarrow^{\mathbf{s}_a}\mathbf{e}\longrightarrow^{\mathbf{s}_b}\mathbf{y}\) (15)

where \(\mathbf{x},\mathbf{y}\in X_\mathrm{AN}\) are node attractor states but **e** exists purely to facilitate the transition. The weights matrix is constructed<sup>2</sup> as

\(\mathbf{W}=\underbrace{\frac{1}{N}\sum_{\mathrm{nodes}\nu}^{N_Z}\mathbf{x}^\nu\mathbf{x}^{\nu\unknown}}_{\mathrm{Hopfieldattractorterms}\longrightarrow}+\underbrace{\frac{1}{N}\sum_{\mathrm{edges}\eta}^{N_E}\mathbf{E}^\eta}_{\mathrm{Asymmetrictransitionterms}\longrightarrow}\) (16)

where \(\mathbf{x}^\nu\in X_\mathrm{AN}\) is the node hypervector corresponding to the \(\nu\)’th node in the graph to be implemented, _N<sub>Z</sub>_ and _N<sub>E</sub>_ are the number of nodes and edges respectively, and \(\mathbf{E}^\eta\) is the addition to the weights matrix required to implement an individual edge, given by

\(\mathbf{E}^{(\eta)}=\mathbf{e}\mathbf{e}^\unknown+H(\mathbf{s}_a)\circ(\mathbf{e}-\mathbf{x})(\mathbf{x}\circ\mathbf{s}_a)^\unknown+H(\mathbf{s}_b)\circ(\mathbf{y}-\mathbf{e})(\mathbf{e}\circ\mathbf{s}_b)^\unknown\) (17)

where **x**, **e** and **y** are the source, edge, and target states of the edge \(\eta\) respectively, and **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_ are the input stimulus hypervectors associated with this edge’s label. The edge index \(\eta\) has been dropped for brevity. The \(\mathbf{ee}\mathbf{e}^\unknown\) term is the edge state attractor we have introduced as an intermediary for the transition. The second set of terms enacts the \(\mathbf{x}\longrightarrow^{\mathbf{s}_a}\mathbf{e}\) transition, by giving a nonzero inner product with the network state **z**_<sub>t</sub>_ only when the network is in state **x**, _and_ the network is being masked by the stimulus **s**_<sub>a</sub>_. When both of these conditions are met, the \((\mathbf{x}\circ\mathbf{s}_a)^\unknown\) term will have a nonzero inner product with the network state, projecting out the (**e**-**x**) term, which "pushes" the network from the **x** to the **e** attractor state. This allows terms to be stored in **W** which are effectively obfuscated, not affecting network dynamics considerably, until a specific stimulus is applied as a mask to the network. Likewise, the third set of terms enacts the \(\mathbf{e}\longrightarrow^{\mathbf{s}_b}\mathbf{y}\) transition.

In the absence of input, the network functions like a standard Hopfield attractor network,

\(\mathbf{Wx}\approx\mathbf{x}\pm\sigma\mathbf{n}\forall\mathbf{x}\in X_\mathrm{AN}\) (18)

where \(\mathbf{n}\in\mathbb{R}^N\) is a standard normally-distributed random vector, and

\(\sigma=\sqrt{\frac{N_Z+3N_E}{N}}\) (19)

is the magnitude of noise due to the undesired finite inner product with other stored terms (see Section VII-A for proof). Thus as long as the magnitude of the noise is not too large, **x** will be a solution of **z**=sgn(**Wz**) and so a fixed-point attractor of the dynamics.

When a valid stimulus is presented as input to the network however, masking the network state, the previously obfuscated asymmetric transition terms become significant and dominate<tbc>

<tbc>the dynamics. Assuming there is a stored transition term **E** corresponding to a valid edge with hypervectors **x**,**e**,**y**,**s**_<sub>a</sub>,_**s**_<sub>b</sub>_ having the same meaning as in Equation 17, during a masking operation we have

\(\mathbf{W}\big(\mathbf{x}\circ H(\mathbf{s}_a)\big)\unknown\underbrace{H\big(\mathbf{s}_a\big)\circ\mathbf{e}}_{\mathrm{Projectiontoedgestate}}+\underbrace{H(-\mathbf{s}_a)\circ\mathbf{x}}_{\mathrm{Maybeignored}}\pm\sqrt{2}\sigma\mathbf{n}\) (20)

where \(\unknown\) implies approximate proportionality (see Section VII-B for proof). The second set of terms can be ignored, as they project only to neurons which are currently being masked. Thus the only significant term is that containing the edge state **e**, which consequently drives the network to the **e** state, enacting the \(\mathbf{x}\longrightarrow^{\mathbf{s}_a}\mathbf{e}\) transition. Since the state **e** is also stored as an attractor within the network, we have<tbc>

\(\mathbf{W}\big(\mathbf{e}\circ H(\mathbf{s}_a)\big)\unknown\mathbf{e}\pm\sqrt{2}\sigma\mathbf{n}\) (21)

<tbc>and<tbc>

\(\mathbf{We}\approx\mathbf{e}\pm\sigma\mathbf{n}\) (22)

<tbc>thus the edge states **e** are also fixed-point attractors of the network dynamics. To complete the transition from state **x** to **y**, the second stimulus **s**_<sub>b</sub>_ is applied, giving

\(\mathbf{W}\big(\mathbf{e}\circ H(\mathbf{s}_b)\big)\unknown H\big(\mathbf{s}_b\big)\circ\mathbf{y}+H(-\mathbf{s}_b)\circ\mathbf{e}\pm\sqrt{2}\sigma\mathbf{n}\) (23)

which drives the network state towards \(\mathbf{y}\in X_\mathrm{AN}\), the desired target attractor state. By consecutive application of the inputs **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_, the transition terms \(\mathbf{E}^\eta\) stored in **W** have thus caused the network to controllably transition from the source state attractor state to the target attractor state. Due to the robustness of the masking mechanism, the stimuli can be applied asynchronously and with arbitrary delays (see Section VII-C). Transition terms \(\mathbf{E}^\eta\) may be iteratively added to **W** to achieve any arbitrary transition between attractor states, and so any arbitrary FSM may be implemented within a large enough attractor network.

## B. Edge outputs

Until now we have not mentioned the other critical component of FSMs: the output associated with every edge. We have separated the construction of transitions and edge outputs for clarity, since the two may be effectively decoupled. Much like for the nodes and edges in the FSM to be implemented, for every unique FSM output \(\rho\in R_\mathrm{FSM}\), we generate a corresponding hypervector \(\mathbf{r}\in R_\mathrm{AN}\), where _R_<sub>AN</sub> is the set of all output hypervectors. We then seek to somehow embed these hypervectors into the attractor network, such that every transition between node attractor states may contain one of these hypervectors **r**. A natural solution would be to embed the **r** hypervector into the edge state attractors \(\mathbf{ee}\mathbf{e}^\unknown\), since there already exists one for every edge. We can consider altering the edge state attractors from \(\mathbf{ee}\mathbf{e}^\unknown\) to \(\mathbf{e}_\mathbf{r}\mathbf{e}_\mathbf{r}^\unknown\), where **e**<sub>**r**</sub> resembles the original **e** state with **r** somehow embedded within it, such that its presence can be detected via a linear projection. If multiple edges have the same **r** hypervector however, then the \(\mathbf{e}_\mathbf{r}\mathbf{e}_\mathbf{r}^\unknown\) terms for different edges will be<tbc>

<sup>2</sup>We have here ignored that the diagonal of **W** is set to 0 (no self connections), but this does not significantly affect the following results.

4

---

## Page 5

correlated, incurring unwanted interference between attractor states and violating the assumption that the inner product between different attractor terms is small enough that it can be ignored. We avoid this by instead storing altered edge state attractors of the form \(\mathbf{e}_\mathbf{r}\mathbf{e}^\unknown\). We then choose **e<sub>r</sub>** such that it is minimally different from **e** (i.e. _d_(**e<sub>r</sub>**,**e**)≈1), so that we still retain the desired attractor dynamics. We thus choose the output hypervectors \(\mathbf{r}\in R_\mathrm{AN}\) to be sparse ternary hypervectors \(\mathbf{r}\in\{-1,0,1\}^N\) with coding level \(f_r:=\frac{1}{N}\sum_i^N|r_i|\), the fraction of nonzero components. These output hypervectors are then embedded in the edge state attractors, altering the \(\mathbf{e}\mathbf{e}^\unknown\) terms in each **E** term according to<tbc>

\(\mathbf{e}\mathbf{e}^\unknown\rightarrow\mathbf{e}_\mathbf{r}\mathbf{e}^\unknown:=\left[\mathbf{e}\circ\big(\mathbf{1}-H(\mathbf{r}\circ\mathbf{r})\big)+\mathbf{r}\right]\mathbf{e}^\unknown\) (24)

<tbc>where the composite vector **e<sub>r</sub>** introduced above is here defined and **1** is a hypervector of all ones. As a result of this modification, the edge states **e** themselves will no longer be exact attractors of the space. The composite state **e<sub>r</sub>** will however be stable, in which the presence of **r** can be easily detected by a linear projection (\(\mathbf{e}_\mathbf{r}\cdot\mathbf{r}=Nf_r\)). This has been achieved without incurring any similarity and thus interference between attractors, which would otherwise alter the dynamics of the previously described transitions. A full transition term \(\mathbf{E}^\eta\), including its output, is thus given by

\(\mathbf{E}^{(\eta)}=\left[\mathbf{e}\circ\big(\mathbf{1}-H(\mathbf{r}\circ\mathbf{r})\big)+\mathbf{r}\right]\mathbf{e}^\unknown+H(\mathbf{s}_a)\circ(\mathbf{e}-\mathbf{x})(\mathbf{x}\circ\mathbf{s}_a)^\unknown+H(\mathbf{s}_b)\circ(\mathbf{y}-\mathbf{e})(\mathbf{e}\circ\mathbf{s}_b)^\unknown\) (25)

which combined with the network state masking operation is solely responsible for storing the FSM connectivity and enabling the desired inter-attractor transition dynamics.

### C. Sparse activity states

It is well known that the memory capacity of attractor networks can be vastly increased by storing sparsely-coded activity patterns, rather than dense patterns as we have done thus far (Amari 1989; Amit 1989; Tsodyks and Feigel’man 1988). We therefore adapt the construction of the attractor network to the case that the network state **z**_<sub>t</sub>_ and its stored hypervectors \(\mathbf{x}^\nu\) are binary and _f_-sparse, i.e. contain mostly zeroes, with very few entries being +1, to test if there are similar gains in the size of FSM that can be reliably embedded. To distinguish these hypervectors from the dense bipolar hypervectors we have been using thus far, we denote sparse binary hypervectors \(\mathbf{x}_\mathrm{sp}\in\{0,1\}^N\) with |**x**<sub>sp</sub>|<sub>1</sub>=_Nf_, where _f_ is the fixed coding level of the states, the fraction of nonzero components. Note that we here construct hypervectors which have _exactly Nf_ nonzero components, and so they may better be described as a sparse _N_-of-_M_ code (Furber et al. 2004). The attractor network’s weights matrix is constructed as

\(\mathbf{W}=\sum_\nu(\mathbf{x}_\mathrm{sp}^\nu-f\mathbf{1})(\mathbf{x}_\mathrm{sp}^\nu-f\mathbf{1})^\unknown+\sum_\eta\mathbf{E}^\eta\) (26)

where \(\mathbf{E}^\eta\) are the equivalent sparse edge terms to be defined. If the neuron state update rule (Equation 10) is replaced with<tbc><tbc>

<tbc>a sparse binary variant, e.g. a top-_k_ activation function or a Heaviside function with an appropriately chosen threshold, then the stored states \(\mathbf{x}_\mathrm{sp}^\nu\) will be attractors of the network’s dynamics (Amari 1989). The additional edge terms \(\mathbf{E}^\eta\) are analogously constructed as

\(\mathbf{E}^{(\eta)}=(\mathbf{e}_\mathrm{sp}-f\mathbf{1})(\mathbf{e}_\mathrm{sp}-f\mathbf{1})^\unknown+(\mathbf{e}_\mathrm{sp}-\mathbf{x}_\mathrm{sp})\big((\mathbf{x}_\mathrm{sp}-f\mathbf{1})\circ\mathbf{s}_a\big)^\unknown+(\mathbf{y}_\mathrm{sp}-\mathbf{e}_\mathrm{sp})\big((\mathbf{e}_\mathrm{sp}-f\mathbf{1})\circ\mathbf{s}_b\big)^\unknown\text{(27)}\)

where the first set of terms embeds the sparse binary edge state **e**<sub>sp</sub> as an attractor, while the second and third terms embed the source-to-edge and edge-to-target transitions respectively. The stimulus hypervectors **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_ can also be made sparse, such that fewer than half of all neurons are masked by the stimuli, but at the cost of decreased memory capacity (Section VII-E). For this reason, we here keep them as bipolar hypervectors, with an approximately equal number of +1 as -1 entries. Each set of terms within each \(\mathbf{E}^\eta\) term performs the same role as in the dense bipolar case as discussed in Section III-A. How output states should be embedded into each transition in the sparse case is unclear, because unlike in the dense case, they cannot be embedded into the edge state attractors without considerably affecting the network dynamics and thus attractor stabilities.

## IV. RESULTS

### A. FSM emulation

To show the generality of FSM construction, we chose to implement a directed graph representing the relationships between gods in ancient Greek mythology, due to the graph’s dense connectivity. The graph and thus FSM to be implemented is shown in Figure 1. From the graph it is clear that a state machine representing the graph must explicitly be capable of state-dependent transitions, e.g. the input "overthrown\_by" must result in a transition to state "Kronos" when in state "Uranus", but to state "Zeus" when in state "Kronos". To construct **W**, the necessary hypervectors are first generated. For every state \(\chi\in X_\mathrm{FSM}\) in the FSM (e.g. "Zeus", "Kronos") a random bipolar hypervector **x** is generated according to Equation 2. For every unique stimulus \(\varsigma\in S_\mathrm{FSM}\) (e.g. "overthrown\_by", "father\_is") a pair of random bipolar stimulus hypervectors **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_ is likewise generated. Similarly, sparse ternary output hypervectors **r** are also generated. The weights matrix **W** is then iteratively constructed as per Equations 16 and 25, with a new hypervector **e** also being generated for every edge. The matrix generated from this procedure we denote **W**<sup>ideal</sup>. For all of the following results, the attractor network is first initialised to be in a certain node attractor state, in this case, "Hades". The network is then allowed to freely evolve for 10 time steps (chosen arbitrarily) as per Equation 10, with every neuron being updated simultaneously on every time step. During this period, it is desired that the network state **z**_<sub>t</sub>_ remains in the attractor state in which it was initialised. An input stimulus **s**_<sub>a</sub>_ is then presented to the network for 10 time steps, during which time the<tbc>

5

---

## Page 6

network state is masked by the stimulus hypervector, and the network evolves synchronously according to Equation 14. If the stimulus corresponds to a valid edge in the FSM, the network state **z**_<sub>t</sub>_ should then be driven towards the correct edge state attractor **e**. After these 10 time steps, the second stimulus hypervector **s**_<sub>b</sub>_ for a particular input is presented for 10 time steps. Again, the network evolves according to Equation 14, and the network should be driven towards the target attractor state **y**, completing the transition. This process is repeated every 30 time steps, causing the network state **z**_<sub>t</sub>_ to travel between node attractor states \(\mathbf{x}\in X_\mathrm{AN}\), corresponding to a valid walk between states \(\chi\in X_\mathrm{FSM}\) in the represented FSM. To view the resulting network dynamics, the similarity between the network state **z**_<sub>t</sub>_ and the edge- and node attractor states is calculated as per Equation 3, such that a similarity of 1 between **z**_<sub>t</sub>_ and some attractor state \(\mathbf{x}^\nu\) implies \(\mathbf{z}_t=\mathbf{x}^\nu\) and thus that the network is inhabiting that attractor. The similarity between the network state **z**_<sub>t</sub>_ and the outputs states \(\mathbf{r}\in R_\mathrm{AN}\) is also calculated, but due to the output hypervectors being sparse, the maximum value that the similarity can take is _d_(**z**_<sub>t</sub>,_**r**)=_f<sub>r</sub>_, which would be interpreted as that output symbol being present.

An attractor network performing a walk is shown in Figure 2, with parameters _N_=10,000, _Nf<sub>r</sub>_=200, _N<sub>Z</sub>_=8, and _N<sub>E</sub>_=16. This corresponds to the network having a<tbc>

<tbc>per-neuron noise (the finite size effect resulting from random hypervectors having a nonzero similarity to each-other) of \(\sigma\approx 0.07\), calculated via Equation 19. The magnitude of the noise is thus small compared with the desired signal of magnitude 1 (Equation 18), and so we are far away from reaching the memory capacity of the network. The network performs the walk as intended, transitioning between the correct node attractor states and corresponding edge states with their associated outputs. The specific sequence of inputs was chosen to show the generality of implementable state transitions. First, there is the explicit state dependence in the repeated input of "father\_is, father\_is". Second, it contains an input stimulus that does not correspond to a valid edge for the currently inhabited state ( "Zeus overthrown\_by"), which should not cause a transition. Third, it contains bidirectional edges ( "consort\_is"), whose repeated application causes the network to flip between two states (between "Kronos" and "Rhea"). And fourthly self-connections, whose target states and source states are identical. Since the network traverses all these edges as expected, we do not expect the precise structure of an FSM’s graph to limit whether or not it can be emulated by the attractor network.

## B. Network robustness

One of the advantages of attractor neural networks that make them suitable as plausible biological models is their robustness to imperfect weights (Amit 1989). That is, individual synapses may have very few bits of precision or become damaged, yet the relevant brain region must still be able to carry out its functional task. To this end, we subjected the network presented here to similar non-idealities, to check that the network retains the feature of global stability and robustness despite being implemented with low-precision and noisy weights. In the first of these tests, the ideal weights matrix \(\mathbf{W}^\mathrm{ideal}\) was binarised and then additive noise was applied, via

\(W_{ij}^\mathrm{noisy}=\mathrm{sgn}\big(W_{ij}^\mathrm{ideal}\big)+\sigma_\mathrm{noise}\cdot\chi_{ij}\) (28)

where \(\chi_{ij}\in\mathbb{R}\) are independently sampled standard Gaussian variables, sampled once during matrix construction, and \(\sigma_\mathrm{noise}\in\mathbb{R}\) is a scaling factor on the strength of noise being imposed. The \(\mathrm{sgn}(\cdot)\) function forces the weights to be bipolar, emulating that the synapses may have only 1 bit of precision, while the \(\chi_{ij}\) random variables act as a smearing on the weight state, emulating that the two weight states have a finite width. A \(\sigma_\mathrm{noise}\) value of 2 thus corresponds to the magnitude of the noise being equal to that of the signal (whether \(W_{ij}^\mathrm{ideal}\ge 0\)), and so, for example, for a damaged weight value of \(W_{ij}^\mathrm{noisy}=+1\) there is a 38% chance that the pre-damaged weight \(W_{ij}^\mathrm{ideal}=-1\). This level of degradation is far worse than is expected even from novel binary memory devices (Xia and Yang 2019), and presumably also for biology. We used the same set of hypervectors and sequence of inputs as in Figure 2, but this time using the degraded weights matrix \(\mathbf{W}^\mathrm{noisy}\), to test the network’s robustness. The results are shown in Figure 3 for weight degradation values of \(\sigma_\mathrm{noise}=2\) and \(\sigma_\mathrm{noise}=5\), corresponding to signal-to-noise ratios (SNRs) of<tbc>

6

Fig. 1: An example FSM which we implement within the attractor network. Each node within the graph (e.g. "Zeus") is represented by a new hypervector \(\mathbf{x}^\mu\) and stored as an attractor within the network. Every edge is labelled by its stimulus (e.g. "father\_is"), for which corresponding hypervectors **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_ are also generated. When a stimulus’ hypervector is input to the network, it should allow all corresponding attractor transitions to take place. Each edge may also have an associated output symbol, where we here choose the edges labelled "type" to output the generation of the god {"Primordial", "Titans", "Olympians"}. This graph was chosen as it displays the generality of the embedding: it contains cycles, loops, bidirectional edges and state-dependent transitions.

---

## Page 7

![picture 1](figures/page_0007_fig_01.png)
*Fig. 2: An attractor network transitioning through attractor states in a state-dependent manner, as a sequence of input stimuli is presented to the network. **a)** The input stimuli to the network, where for each unique stimulus (e.g. "father\_is" in the FSM to be implemented (Figure 1) a pair of hypervectors **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_ have been generated. No stimulus, a stimulus **s**_<sub>a</sub>_, then a stimulus **s**_<sub>b</sub>_ are input for 10 time steps each in sequence. **b) & c)** The similarity of the network state **z**_<sub>t</sub>_ to stored node attractor states \(\mathbf{x}\in X_\mathrm{AN}\) and stored edge states **e** respectively, computed via the inner product (Equation 3). **d)** The similarity of the network state **z**_<sub>t</sub>_ to the sparse output states \(\mathbf{r}\in R_\mathrm{AN}\). All similarities have been labelled with the state they represent and the colours are purely illustrative. The attractor transitions shown here are explicitly state-dependent, as can be seen from the repeated input of the stimulus "father\_is", which results in a transition to state "Kronos" when in "Hades", but to "Uranus" when in "Kronos". Additionally, the network is unaffected by nonsense input that does not correspond to a stored edge, as the network remains in the attractor "Uranus" when presented with the stimulus "father\_is".*

<tbc>0dB and -0.8dB respectively. We see that for \(\sigma_\mathrm{noise}=2\) the attractor network performs the walk just as well as in Figure 2, which used the ideal weights matrix, despite the fact that here the binary weight distributions overlap each-other considerably. Furthermore, we have that \(d(\mathbf{z}_t,\mathbf{x}^\nu)\approx 1\) where \(\mathbf{x}^\nu\) is the attractor that the network should be inhabiting at any time, indicating that the attractor stability and recall accuracy is unaffected by the non-idealities. For \(\sigma_\mathrm{noise}=5\), a scenario where the realised weight carries very little information about the ideal weight’s value, we see that the network nonetheless continues to function, performing the correct walk between attractor states. However, there is a degradation in the recall of stored attractor states, with the network state no longer converging to a similarity of 1 with the stored attractor states. For greater values of \(\sigma_\mathrm{noise}\), the network ceases to perform the correct walk, and indeed does not converge on any stored attractor state (not shown).

A further test of robustness was to restrict the weights matrix to be sparse, as a dense all-to-all connectivity may not be feasible in biology, where synaptic connections are spatially constrained and have an associated chemical cost. Similar to<tbc>

<tbc>the previous test, the sparse weights matrix was generated via<tbc>

\(W_{ij}^\mathrm{sparse}=\mathrm{sgn}(W_{ij}^\mathrm{ideal})\cdot H(|W_{ij}|-\theta)\) (29)

<tbc>where \(\theta\) is a threshold set such that \(\mathbf{W}^\mathrm{sparse}\in\{-1,0,1\}^{N\times N}\) has the desired sparsity. Through this procedure, only the most extreme weight values are allowed to be nonzero. Since the terms inside **W**<sup>ideal</sup> are symmetrically distributed around 0, there are approximately as many +1 entries in **W**<sup>sparse</sup> as - 1s. Using the same hypervectors and sequence of inputs as before, an attractor network performing a walk using the sparse weights matrix **W**<sup>sparse</sup> is shown in Figure 4, with sparsities of 98% and 99%. We see that for the 98% sparse case, there is again very little difference with the ideal case shown in Figure 2, with the network still having a similarity of \(d(\mathbf{z}_t,\mathbf{x})\approx 1\) with stored attractor states, and performing the correct walk. When the sparsity is pushed further to 99% however, we see that despite the network performing the correct walk, the attractor states are again slightly degraded, with the network converging on states with \(d(\mathbf{z}_t,\mathbf{x}^\nu)<1\) with stored attractor states \(\mathbf{x}^\nu\). For greater sparsities, the network ceases to perform the correct walk, and again does<tbc>

7

---

## Page 8

![picture 1](figures/page_0008_fig_01.png)

![picture 2](figures/page_0008_fig_02.png)
*Fig. 4: The attractor network performing a walk as in Figure 2, but using a sparse ternary weights matrix \(\mathbf{W}^\mathrm{sparse}\in\{-1,0,1\}^{N\times N}\), generated via Equation 29. The weights matrices for **a)** and **b)** are 98% and 99% sparse respectively. Shown are the similarities of the network state **z**_<sub>t</sub>_ with stored node hypervectors \(\mathbf{x}\in X_\mathrm{AN}\), with the applied stimulus hypervector at any time shown above. We see that even when 98% of the entries in **W** are zeroes, the network continues to function with negligible loss in stability, as the correct walk between attractor states is performed, and the network converges on stored attractors with similarity _d_(**z**_<sub>t</sub>,_**x**)≈1. At 99% sparsity there is a degradation in the accuracy of stored attractors, with the network converging on states with _d_(**z**_<sub>t</sub>,_**x**)_\<_1, but with the correct walk still being performed. Beyond 99% sparsity the attractor dynamics break down (not shown). Thus although requiring a large number of neurons _N_ to enforce state pseudo- orthogonality, the network requires far fewer than _N_<sup>2</sup> nonzero weights to function robustly.*

8

---

## Page 9

not converge on any stored attractor state (not shown).

These two tests thus highlight the extreme robustness of the model to imprecise and unreliable weights. The network may be implemented with 1 bit precision weights, whose weight distributions are entirely overlapping, or set 98% of the weights to 0, and still continue to function without any discernible loss in performance. The extent to which the weights matrix may be degraded and the network still remain stable is of course a function not only of the level of degradation, but also of the size of the network _N_, as well as the the number of FSM states _N<sub>Z</sub>_ and edges _N<sub>E</sub>_ stored within the network. For conventional Hopfield models with Hebbian learning, these two factors are normally theoretically treated alike, as contributing an effective noise to the postsynaptic sum as in Equation 19, and so the magnitude of withstandable synaptic noise increases with increasing _N_ (Amit 1989; Sompolinsky 1987). Although a thorough mathematical investigation into the scaling of weight degradation limits is justified, as a first result we have here given numerical data showing stability even in the most extreme cases of non-ideal weights, and expect that any implementation of the network with novel devices would be far away from such extremities.

## C. Asynchronous updates

Another useful property of Hopfield networks is the ability to robustly function even with asynchronously updating neurons, wherein not every neuron experiences a simultaneous state update. This property is especially important for any architecture claiming to be biologically plausible, as biological neurons update asynchronously and largely independent of each-other, without the the need for global clock signals. To this end, we ran a similar experiment to that in Figure 2, using the undamaged weights matrix **W**ideal, but with an asynchronous neuron update rule, wherein on each time step every neuron has only a 10% chance of updating its state. The remaining 90% of the time, the neuron retains its state from the previous time step, regardless of its postsynaptic sum. There is thus no fixed order of neuron updates, and indeed it is not even a certainty that a neuron will update in any finite time. To account for the slower dynamics of the network state, the time for which inputs were presented to the network, as well as the periods without any input, was increased from 10 to 40 time steps. To be able to easily view the gradual state transition, three of the node hypervectors were chosen to be columns of the _N_-dimensional Hadamard matrix, rather than being randomly generated. The results are shown in Figure 5, for a shorter sequence of stimulus inputs. We see that the network functions as intended, but with the network now converging on the correct attractors in a finite number of updates rather than in just one. The model proposed here is thus not reliant on synchronous dynamics, which is important not only for biological plausibility, but also when considering possible implementations on asynchronous neuromorphic hardware (Davies et al. 2018; Liu et al. 2014).<tbc>

## D. Storage capacity

It is well known that the storage capacity of a Hopfield network, the number of patterns _P_ that can be stored and reliably retrieved, is proportional to the size of the network, via _P\<_0.14_N_ (Amit 1989; Hopfield 1982). When one tries to store more than _P_ attractors within the network, the so- called memory blackout occurs, after which no pattern can be retrieved. We thus perform numerical simulations for a large range of attractor network and FSM sizes, to see if an analogous relationship exists. Said otherwise, for an attractor network of finite size _N_, what sizes of FSM can the network successfully emulate?

For a given _N_, number of FSM states _N<sub>Z</sub>_ and edges _N<sub>E</sub>_, a random FSM was generated and an attractor network constructed to represent it as described in Section III. To ensure a reasonable FSM was generated, the FSM’s graph was first generated to have all nodes connected in a sequential ring structure, i.e. every state \(\chi^\nu\in X_\text{FSM}\) connects to \(\chi^{\nu+1\mathrm{mod}N_Z}\). The remaining edges between nodes were selected at random, until the desired number of edges _N<sub>E</sub>_ was reached. For each edge an associated stimulus is then required. Although one option would be to allocate as few unique stimuli as possible, so that the state transitions are maximally state-dependent, this results in some advantageous cancellation effects between the \(\mathbf{E}^\eta\) transition terms and the stored attractors \(\mathbf{x}^\nu\mathbf{x}^{\nu\mathrm{r}}\). To instead probe a worst-case scenario, each edge was assigned a unique stimulus.

With the FSM now generated, an attractor network with _N_ neurons was constructed as previously described. An initial attractor state was chosen at random, and then a random valid walk between states was chosen to be performed (chosen arbitrarily to be of length 6, corresponding to each run taking 180 time steps). The corresponding sequence of stimuli was input to the attractor network via the same procedure as in Figure 2, each masking the network state in turn. Each run was then evaluated to have either passed or failed, with a pass meaning that the network state inhabited the correct attractor state with overlap \(d(\mathbf{z}_t,\mathbf{x}^\nu)>0.5\) in the middle of all intervals when it should be in a certain node attractor state. This 0.5-criterion was chosen since, for a set of orthogonal hypervectors, at most only one hypervector may satisfy the criterion at once. A pass thus corresponds to the network performing the correct walk between attractor states. The results are shown in Figure 6. We see that for a given _N_, there is a linear relationship between the the number of nodes _N<sub>Z</sub>_ and number of edges _N<sub>E</sub>_ in the FSM that can be implemented before failure. That this trade- off exists is not surprising, since both contribute additively to the SNR within the attractor network (Equation 19). For each _N_, a linear Support Vector Machine (SVM) was fitted to the data, to find the separating boundary at which failure and success of the walk are approximately equiprobable. The boundary is given by \(N_Z+\beta N_E=c(N)\), where \(\beta\) represents the relative cost of adding nodes and edges, and _c(N_) is an offset. For all of the fitted boundaries, the value of \(\beta\) was found to be approximately constant, with \(\beta=2.2\pm 0.1\), and so is<tbc>

9

---

## Page 10

![picture 1](figures/page_0010_fig_01.png)
*Fig. 5: An attractor network performing a shorter walk than in Figure 2, but where neurons are updated asynchronously, with each neuron having a 10% chance of updating on any time step. **a)** The similarity of the network state **z**_<sub>t</sub>_ to stored node hypervectors, with the stimulus hypervectors being applied to the network labelled above. **b)** The evolution of a subset of neurons within the attractor network, where for visual clarity, three of the node hypervectors have been taken from columns of the _N_-dimensional Hadamard matrix, rather than being randomly generated. The network functions largely the same as in the synchronous case, but with transitions between attractor states now taking a finite number of time steps to complete. The model is thus not dependent on the precise timing of neuron updates, and should function robustly in asynchronous systems where timing is unreliable.*

<tbc>assumed to be independent of _N_. For every value of _N_, we define the capacity _C_ to be the maximum size of FSM which can be implemented before failure, for which _N<sub>E</sub>=N<sub>Z</sub>_. The capacity _C_ is then given by \(C(N)=\frac{c(N)}{1+\beta}\) , and is also plotted in Figure 6. A linear fit reveals an approximate proportionality relationship of _C(N_)≈0.029_N_. Combining these two results, the boundary which limits the size of FSM which can be emulated is then given by

\(N_Z+2.2N_E<0.10N\) (30)

It is expected that additional edges consume more of the network’s storage capacity than additional nodes, since for every edge, 5 additional terms are added to **W** (Equation 25), contributing 3\(\times\) as much cross-talk noise as adding a node would (Equation 19). We can compare this storage capacity relation with that of the standard Hopfield model, by considering the case _N<sub>E</sub>_=0, i.e. there are no transition terms in the network, and so the network is identical to a standard Hopfield network. In this case, our failure boundary would become _N<sub>Z</sub>\<_0.10_N_, in comparison to Hopfield’s _P\<_0.14_N_.

## E. Storage capacity with sparse states

The same FSM as shown in Figure 1 was embedded into an attractor network via the construction scheme described in Section III-C, with values _N_=10,000 neurons and coding level _f_=0.1. To enforce the correct sparsity in the neural state, the \(\mathrm{sgn}(\cdot)\) activation function (Equation 10) was replaced with a top-_k_ activation function (also known as "_k_- Winners-Take-All")

\(\mathbf{z}_{t+1,\mathrm{sp}}=H\big(\mathbf{W}\mathbf{z}_{t,\mathrm{sp}}-\theta\big)\) (31)

where \(H(\cdot)\) is a component-wise Heaviside function, and \(\theta\) is chosen to be the _Nf_’th largest value of **W**z**<sub>_t,_sp</sub>, to enforce that **z**<sub>_t_+1,sp</sub> is _f_-sparse. While a stimulus hypervector \(\mathbf{s}\in\{-1,1\}^N\) is being applied as a mask to the network, the activation function is similarly

\(\mathbf{z}_{t+1,\mathrm{sp}}=H\big(\mathbf{W}(\mathbf{z}_{t,\mathrm{sp}}\circ H(\mathbf{s}))-\theta\big)\) (32)

with \(\theta\) being chosen in the same manner. Note that although the introduction of this adaptive \(\theta\) threshold mechanism may seem to be somewhat biologically implausible, or at least a tall order for any possible neural implementation, it may easily be implemented using a suitably connected population of inhibitory feedback neurons, which silence all attractor neurons except those that receive the greatest input (Amari 1989; Lin et al. 2014). The sparse attractor network is shown performing a walk between the correct attractor states in Figure 7, as a sequence of stimuli is applied as input to the network. In contrast to the dense bipolar case, the maximum overlap between the network state **z**<sub>_t,_sp</sub> and a stored attractor state \(\mathbf{x}_\mathrm{sp}^\nu\) is now \(d(\mathbf{z}_{t,\mathrm{sp}},\mathbf{x}_\mathrm{sp}^\nu)=f=0.1\), while the expected overlap between unrelated states is _f_<sup>2</sup>=0.01 rather than 0.

We now apply the same procedure as in the dense case for determining the memory capacity of the sparse-activity attractor network. For direct comparison with the dense case, we define the memory capacity _C(N_) to be the largest FSM with _N<sub>E</sub>=N<sub>Z</sub>_ for which walk success and failure are equiprobable. For every tested (_N,f,N<sub>Z</sub>_) tuple we generate a corresponding set of hypervectors and weights matrix as discussed in Section III-C, and then randomly choose a walk between 6 node attractor states to be completed. The chosen<tbc>

10

---

## Page 11

![picture 1](figures/page_0011_fig_01.png)

![picture 2](figures/page_0011_fig_02.png)
*Fig. 7: The attractor network performing a walk between sparse attractor states, where the neurons have a top-_k_ binary activation function to enforce the desired sparsity (Equations 31 & 32), and the weights matrix is constructed as discussed in Section III-C. The values used here are _N_=10,000 neurons with coding level _f_=0.1, such that in any sparse hypervector 1000 components are +1 while the rest are 0. **a)** The input stimuli to the network, consisting of dense bipolar hypervectors applied as multiplicative masks. **b)** & **c)** The overlap between the network state **z**<sub>_t,_sp</sub> to stored node attractor states **x**<sub>sp</sub> and stored edge attractor states **e**<sub>sp</sub> respectively, computed via the inner product (Equation 3). Note that since the network and attractor states are now sparse binary, the maximum possible overlap value is _f_=0.1, while independently generated states have an expected overlap of _f_<sup>2</sup>=0.01*

11

---

## Page 12

<tbc>walk then determines the sequence of stimuli to be input, and each stimulus is then applied for 10 time steps. Each (_N,f,N<sub>Z</sub>_) tuple was then determined to have passed or failed, with a success criterion that \(d(\mathbf{z}_{t,\text{sp}},\mathbf{x}_\text{sp}^\nu)>\frac{1}{2}(f+f^2)\) in the middle of all intervals when the network should be in a certain node attractor state. This criterion was chosen as it is the sparse analogue of that used in the dense case: at most only one attractor state may satisfy it at any time.

The results are shown in Figure 8. We see that for a fixed number of neurons _N_, the size of FSM that may be stored initially increases as _f_ is decreased, but below a certain _f_ value drops off rapidly. To estimate the optimal coding level _f_ and maximum FSM size _N<sub>Z</sub>_ for an attractor network of size _N_, we apply a 2D Gaussian convolutional filter with standard deviation 3 over the grid of successes/failures for each _N_ value separately, in order to obtain a kernel density estimate (KDE) _p_KDE of the walk success probability. The capacity _C(N_) was then obtained by taking the maximum _N<sub>Z</sub>_ value for which \(p_{\mathrm{KDE}}\ge 0.5\). This procedure was chosen in order to be comparable to that performed in the dense bipolar case (Figure 6), where a linear separation boundary between success and failure was used instead. Plotting capacity _C_ against _N_ and applying a linear fit in the log-log domain reveals a scaling relation of _C∼N_<sup>1.90</sup>. This approximately quadratic scaling in the sparse case is a vast improvement over the linear scaling shown in the dense case (Figure 6), and is in keeping with the theoretical scaling estimates of _P_max∼_N_<sup>2</sup>/(log_N_)<sup>2</sup> for sparsely-coded binary attractor networks (Amari 1989). The optimal coding level _f_ is also shown, and a linear fit in the log-log domain implies a scaling relation of the form _f∼N_<sup>-0.949</sup>. Again, this is similar to the theoretically optimal _f(N_) scaling relation for sparse binary attractor networks, where the coding level scales like _f_∼(log_N)/N_ (Amari 1989).

## V. RELATION TO OTHER ARCHITECTURES

### A. FSM emulation

While there is a large body of work concerning the equivalence between RNNs and FSMs, their implementations broadly fall into a few categories. There are those that require iterative gradient descent methods to mimic an FSM (Das and Mozer 1994; Lee Giles et al. 1995; Pollack 1991; Zeng et al. 1993), which makes them difficult to train for large FSMs, and improbable for use in biology. There are those that require creating a new FSM with an explicitly expanded state set, \(Z':=Z\times S\), such that there is a new state for every old state- stimulus pair (Alquézar and Sanfeliu 1995; Minsky 1967), which is unfavourable due to the the explosion of (usually one-hot) states needing to be represented, as well as the difficulty of adding new states or stimuli iteratively. There are those that require higher-order weight tensors in order to explicitly provide a weight entry for every unique state- stimulus pair (Forcada and Carrasco 2001; Mali et al. 2020; Omlin et al. 1998) which, as well as being non-distributed, may be more difficult to implement, for example requiring the use of Sigma-Pi units (Groschner et al. 2022; Koch 1998)<tbc>

<tbc>or a large number of hidden neurons with 2-body synaptic interactions only (Krotov and Hopfield 2021).

In Recanatesi et al. 2017 transitions are triggered by adiabatically modulating a global inhibition parameter, such that the network may transition between similar stored patterns. Lacking however is a method to construct a network to perform arbitrary, controllable transitions between states. In Chen and Miller 2020 an in-depth analysis of small populations of rate-based neurons is conducted, wherein synapses with short- term synaptic depression enable a rich behaviour of itinerancy between attractor states, but does not scale to large systems and arbitrary stored memories.

Most closely resembling our approach, however, are earlier works concerned with the related task of creating a sequence of transitions between attractor states in Hopfield-like neural networks. The majority of these efforts rely upon the use of synaptic delays, such that the postsynaptic sum on a time step _t_ depends, for example, also on the network state at time _t_-10, rather than just _t_-1. These delay synapses thus allow attractor cross-terms of the form \(\mathbf{x}^{\nu+1}\mathbf{x}^{\nu\unknown}\) to become influential only after the network has inhabited an attractor state for a certain amount of time, triggering a walk between attractor states (Kleinfeld 1986; Sompolinsky and Kanter 1986). This then also allowed for the construction of networks with state- dependent input-triggered transitions (Amit 1988; Drossaers 1992; Gutfreund and Mezard 1988). Similar networks were shown to function without the need for synaptic delays, but require fine tuning of network parameters and suffer from extremely low storage capacity (Amit 1989; Buhmann and Schulten 1987). In any case, the need for synaptic delay elements represents a large requirement on any substrate which might implement such a network, and indeed are problematic to implement in neuromorphic systems (Nielsen et al. 2017).

State-dependent computation in spiking neural networks was realised in Neftci et al. 2013 and Liang et al. 2019, where they used population attractor dynamics to achieve robust state representations via sustained spiking activity. Additionally, these works highlight the need for robust-yet-flexible neural state machine primitives, if one is to succeed in designing intelligent end-to-end neuromorphic cognitive systems. These approaches differ from this work however in that the state representations are still fundamentally population-based rather than distributed, and so pose difficulties such as the requirement of finding a new population of neurons to represent any new state (Rutishauser and Douglas 2009).

In Rigotti et al. 2010 they discuss the need for a mechanism to induce flips in the neuron state (i.e. an operation akin to a Hadamard product) in order to directly implement nontrivial switching between different attractor states, but disqualify such a mechanism from plausibly existing using synaptic currents alone. We also reject such a mechanism as a biologically plausible solution, but on the grounds that it would not robustly function in an asynchronous neural system (see Section VII-C). They instead show the necessity of a population of neurons with mixed selectivity, connected to both the input and attractor neurons, in order to achieve the desired attractor<tbc>

12

---

## Page 13

![picture 1](figures/page_0013_fig_01.png)
*Fig. 8: The capacity of the attractor network with sparse binary activity and attractor states, for varying coding level _f_. **a)** Each coloured square is a successful walk, with no unique (_N,f,N<sub>Z</sub>_) tuple being tested more than once, and lower-_N_ squares occlude higher-_N_ squares for visual clarity. To comply with the definition of the memory capacity _C_ in the dense case, each FSM was generated with an equal number of states as edges, _N<sub>Z</sub>=N<sub>E</sub>_. The capacity _C_ is taken as the maximum _N<sub>Z</sub>_ value for an _N_ at which the walk success probability \(p_\mathrm{KDE}\ge 50\%\), estimated via a Gaussian KDE and indicated by the black crosses. **b)** The capacities _C_ obtained by this procedure for varying attractor network sizes _N_, up to _N_=40,000, and **c)** the coding levels _f_ at these points. Linear fits are overlain for each, implying an approximately quadratic scaling relation for the memory capacity _C∼N_<sup>1.90</sup> and an approximately inverse scaling relation for the coding level _f∼N_<sup>-0.949</sup>.*

itinerancy dynamics. This requirement arose by demanding that the network state switch to resembling the target state immediately upon receiving a stimulus. We instead show that similar results can be achieved without this extra population, if we relax to instead demanding only that the network soon evolve to the target state.

The main contribution of this article is thus to introduce a method by which attractor networks may be endowed with state-dependent attractor-switching capabilities, without requiring biologically implausible elements or components which are expensive to implement (e.g. precise synaptic delays), and can be scaled up efficiently. The extension to arbitrary FSM emulation shows the generality of the method, and that its limitations can be overcome by the appropriate modifications, like introducing the edge state attractors (Section VII-D).

## B. VSA embeddings

This work also differs from more conventional methods to implement graphs and FSMs in VSAs (Kleyko, Rachkovskij, et al. 2022; Osipov et al. 2017; Poduval et al. 2022; Teeters et al. 2023; Yerxa et al. 2018), in that the network state does not need to be read by an outsider in order to implement the state transition dynamics. That is, where in previous works a graph is encoded by a hypervector (or an associative memory composed of hypervectors) such that the desired dynamics and outputs may be reliably decoded by external circuitry, we instead encode the graph’s connectivity within the attractor network’s weights matrix, such that its recurrent neural dynamics realise the desired state machine behaviour.

The use of a Hopfield network as an auto-associative cleanup memory in conjunction with VSAs has been explored in previous works, including theoretical analyses of their capacity to store bundled hypervectors with different representations (Clarkson et al. 2023), and using single attractor states to retrieve knowledge structures from partial cues (Steinberg and Sompolinsky 2022). Further links between VSAs and attractor networks have also been demonstrated with the use of complex phasor hypervectors - rather than binary or bipolar hypervectors - being stored as attractors within phasor neural networks (Frady and Sommer 2019; Kleyko, Rachkovskij, et al. 2022; Noest 1987; Plate 2003). Complex phasor hypervectors are of particular interest in neuromorphic computing, since they may be very naturally implemented with spike-timing phasor codes, wherein the value represented by a neuron is encoded by the precise timing of its spikes with respect to other neurons or a global oscillatory reference signal, and hypervector binding may be implemented by phase addition (Auge et al. 2021; Orchard and Jarvis 2023).

In Osipov et al. 2017 the authors show the usefulness of VSA representations for synthesizing state machines from observable data, which might be combined with this work to realise a neural system that can synthesise appropriate attractor itinerancy dynamics to best fit observed data. Similarly, if equally robust attractor-based neural implementations of other primitive computational blocks could be created - such as a stack - then they might be combined to create more complex VSA-driven cognitive computational structures, such as neural Turing machines (Graves et al. 2014; Grefenstette et al. 2015; Yerxa et al. 2018). Looking further, this combined with the end-to-end trainability of VSA models could pave the way for<tbc>

13

---

## Page 14

neural systems which have the explainability, compositionality and robustness thereof, but the flexibility and performance of deep neural networks (Hersche et al. 2023; Schlag et al. 2020).

## VI. BIOLOGICAL PLAUSIBILITY

Transitions between discrete neural attractor states are thought to be a crucial mechanism for performing context- dependent decision making in biological neural systems (Daelli and Treves 2010; Mante et al. 2013; Miller 2016; Tajima et al. 2017). Attractor dynamics enable a temporary retention of received information, and ensure that irrelevant inputs do not produce stable deviations in the neural state. Such networks are widely theorised to exist in the brain, for example in the hippocampus for its pattern completion and working memory capabilities (Khona and Fiete 2022; Rolls 2013). As such, we showed that a Hopfield attractor network and its sparse variant can be modified such that they can perform stimulus-triggered state-dependent attractor transitions, without resorting to additional biologically-implausible mechanisms and while abiding by the principles of distributed representation. The changes we introduced are a) an altered weights matrix construction with additional asymmetric cross- terms (which does not incur any considerable extra complexity) and b) the ability for a stimulus to mask a subset of neurons within the attractor population. As long as such a mechanism exists, the network proposed here could thus map onto brain areas theorised to support attractor dynamics. The masking mechanism could, for example, feasibly be achieved by a population of inhibitory neurons representing the stimuli, which selectively project to neurons within the attractor population.

### A. Robustness

The robust functioning of the network despite noisy and unreliable weights is a crucial prerequisite for the model to plausibly be able to exist in biological systems. As we have shown, the network weights may be considerably degraded without affecting the behaviour of the network, and indeed beyond this the network exhibits a so-called graceful degradation in performance. Furthermore, biological synapses are expected to have only a few bits of precision (Baldassi et al. 2016; Bartol et al. 2015; O’Connor et al. 2005), and the network has been shown to function even in the worst case of binary weights. These properties stem from the massive redundancy arising from storing the attractor states across the entire synaptic matrix in a distributed manner, a technique that the brain is expected to utilise (Crawford et al. 2016; Rumelhart and McClelland 1986). Of course, we expect there to be a trade-off between the amount of each non-ideality that the network can withstand before failure. That is, an attractor network with dense noisy weights may withstand a greater degree of synaptic noise than if the weights matrix were also made sparse. Likewise, larger networks storing the same sized FSM should be able to withstand greater non-idealities than smaller networks, as is the case for attractor networks in general (Amit 1989; Sompolinsky 1987).

Since the network is still an attractor network, it retains all of the properties that make them suitable for modelling cognitive function, such as that the network can perform robust pattern completion and correction, i.e. the recovery of a stored prototypical memory given a damaged, incomplete or noisy version, and thereafter function as a stable working memory (Amit 1989; Hopfield 1982).

The robustness of the network to weight non-idealities also makes it a prime candidate for implementation with novel memristive crossbar technologies, which would allow an efficient and high-density implementation of the matrix-vector multiplication required in the neural state update (Equation 14) to be performed in one operation (Ielmini and Wong 2018; Verleysen and Jespers 1989; Xia and Yang 2019). Akin to the biological synapses they emulate, such devices also often have only a few bits of precision, and suffer from considerable per-device mismatch in the programmed conductance states. The network proposed in this article is thus highly suitable for implementation with such architectures, as we have shown that robust performance is retained even when the network is subjected to very high degree of such non-idealities.

The continued functionality of the network when its dynamics are asynchronous is another important factor when considering its biological plausibility. In a biological neural system, neurons will produce action potentials whenever their membrane potential happens to exceed the neuron’s spiking threshold, rather than all updating synchronously at fixed time intervals. We tested the regime where the timescale of the neuron dynamics is much slower than the timescale of the input, by replacing the synchronous neuron update rule with a stochastic asynchronous variant thereof, and showed that the network is robust to this asynchrony. Similarly, we tested the regime where neuron dynamics are much faster than the input, by considering input which is applied stochastically and asynchronously instead (Section VII-C). The continued robustness of the model in these two extreme asynchronous regimes implies that the network is dependent neither upon the exact timing of inputs to the network, nor on the neuron updates within the network, and so would function robustly both in biological neural systems and asynchronous neuromorphic systems where the exact timing of events cannot be guaranteed (Davies et al. 2018; Liu et al. 2014).

### B. Learning

The procedure for generating the weights matrix **W**, as a result of its simplicity, makes the proposed network more biologically plausible than other more complex approaches, e.g. those utilising gradient descent methods. It can be learned in one-shot in a fully online fashion, since adding a new node or edge involves only an additive contribution to the weights matrix, which does not require knowledge of irrelevant edges, nodes, their hypervectors, or the weight values themselves. Furthermore, as a result of the entirely distributed representation of states and transitions, new behaviours may be added to the weights matrix at a later date, both without having to allocate new hardware, and without having to recalculate **W**<tbc>

14

---

## Page 15

<tbc>with all previous data. Both of these factors are critical for continual online learning.

Evaluating the local learnability of **W** to implement transitions is also necessary to evaluate the biological plausibility of the model. In the original paper by Hopfield, the weights could be learned using the simple Hebbian rule<tbc>

\(\delta w_{ij}=x_i^\nu x_j^\nu\) (33)

<tbc>where \(x_i^\nu\) and \(x_j^\nu\) are the activities of the post- and presynaptic neurons respectively, and \(\delta w_{ij}\) the online synaptic efficacy update (Hebb 1949; Hopfield 1982). While the attractor terms within the network can be learned in this manner, the transition cross-terms that we have introduced require an altered version of the learning rule. If we simplify our network construction by removing the edge state attractors, then the local weight update required to learn a transition between states is given by

\(\delta w_{ij}=H(s_i)y_ix_js_j\) (34)

where **y**, **x** and **s** are as previously defined. In removing the edge states, we disallow FSMs with consecutive edges with the same stimulus (e.g. "father\_is, father\_is"), but this is not a problem if completely general FSM construction is not the goal per se (see Section VII-D, Figure 12). This state-transition learning rule is just as local as the original Hopfield learning rule, as the weight update from presynaptic neuron _j_ to postsynaptic neuron _i_ is dependent only upon information that may be made directly accessible in the pre- and postsynaptic neurons, and does not depend on information in other neurons to which the synapse is not connected (Khaef et al. 2022; Zenke and Neftci 2021).

From the hardware perspective, the locality of the learning rule means that if the matrix-vector multiplication step in the neuron state update rule is implemented using novel memristive crossbar circuits (Ielmini and Wong 2018; Xia and Yang 2019; Zidan and Lu 2020), then the weights matrix could be learned online and in-memory via a sequence of parallel conductance updates, rather than by computing the weights matrix offline and then writing the summed values to the devices’ conductances. As long as the updates in the memristors’ conductances are sufficiently linear and symmetric, then attractors and transitions could be sequentially learned in one-shot and in parallel by specifying the two hypervectors in the outer product weight update at the crossbar’s inputs and outputs by appropriately shaped voltage pulses (Alibart et al. 2013; Li et al. 2021).

## C. Scaling

When the FSM states are represented by dense bipolar hypervectors within the attractor network, we found a linear scaling between the size of the network _N_ and the capacity _C_ in terms of the size of FSM that could be embedded without errors. Although this is in keeping with the results in the Hopfield paper, this is not a favourable result when considering the biological plausibility of the system for large _N_ (Hopfield 1982). Since the attractor network is fully connected, the<tbc>

<tbc>capacity actually scales sublinearly \(C\sim\sqrt{N_\mathrm{syn}}\) with the number of synapses _N_<sub>syn</sub>, meaning that an increasing number of synapses are required per attractor and transition to be stored for large _N_, and so the network becomes increasingly inefficient. Additionally, the fact that every neuron is active at any time (or half of them, depending on interpretation of the -1 state) represents an unnecessarily large energy burden for any system utilising this model. This is in contrast to data from neural recordings, where a low per-neuron mean activity is ensured by the sparse coding of information (Barth and Poulet 2012; Olshausen and Field 2004; Rolls and Treves 2011).

We thus tested how the capacity of the network scales with _N_ when the FSM states are instead represented by sparse binary hypervectors with coding level _f_, since it is well known that the number of sparse binary vectors that can be stored in an attractor network scales much more favourably, _P∼N_<sup>2</sup>/(log_N_)<sup>2</sup> (Amari 1989). We found indeed that the sparse coding of the FSM states vastly improved the capacity of the network, scaling approximately quadratically with _C∼N_<sup>1.90</sup>, and so approximately linearly in the number of synapses. This linear scaling with the number of synapses ensures not only the efficient use of available synaptic resources in biological systems, but is especially important when one considers a possible implementation in neuromorphic hardware, where the number of synapses usually represents the main size constraint, rather than the number of neurons (Davies et al. 2018; Manohar 2022).

The coding level _f_ was found to have an approximately inverse relationship with the attractor network size, _f∼N_<sup>-0.949</sup>, which would imply that the number of active neurons _Nf_ in any attractor state grows very slowly, _Nf∼N_<sup>0.051</sup>. This is in agreement with the theoretically optimal case, where the coding level for a sparse binary attractor network should scale like _f_∼(log_N)/N_, and so the number of active neurons in any pattern scales like _Nf_∼log_N_ (Amari 1989).

Sparsity in the stored hypervectors is especially important when one considers how the weights matrix **W** could be learned in an online fashion, if the synapses are restricted to have only a few bits of precision. So far we have considered quantisation of the weights only _after_ the summed values have been determined, whereas including weight quantisation while new patterns are being iteratively learned is a much harder problem, and implies attractor capacity relations as poor as _P_∼log_N_. One solution is for the states to be increasingly sparse, in which case the optimal scaling of _P∼N_<sup>2</sup>/(log_N_)<sup>2</sup> can be recovered (Amit and Fusi 1994; Brunel et al. 1998).

In short, by letting the FSM states be represented by sparse binary hypervectors rather than dense bipolar hypervectors, we not only move closer to a more biologically realistic model of neural activity, but also benefit from the superior scaling properties of sparse binary attractor networks, which lets the maximum size of FSM that can be embedded scale approximately quadratically with the attractor network size rather than linearly.

15

---

## Page 16

## VII. CONCLUSION

Attractor neural networks are robust abstract models of human memory, but previous attempts to endow them with complex and controllable attractor-switching capabilities have suffered mostly from being either non-distributed, not scalable, or not robust. We have here introduced a simple procedure by which any arbitrary FSM may be embedded into a large- enough Hopfield-like attractor network, where states and stimuli are represented by high-dimensional random hypervectors, and all information pertaining to FSM transitions is stored in the network’s weights matrix in a fully distributed manner. Our method of modelling input to the network as a masking of the network state allows cross-terms between attractors to be stored in the weights matrix in a way that they are effectively obfuscated until the correct state-stimulus pair is present, much in a manner similar to the standard binding- unbinding operation in more conventional VSAs.

We showed that the network retains many of the features of attractor networks which make them suitable for biology, namely that the network is not reliant on synchronous dynamics and is robust to unreliable and imprecise weights, thus also making it highly suitable for implementation with high-density but noisy devices. We presented numerical results showing that the network capacity in terms of implementable FSM size scales linearly with the size of the attractor network for dense bipolar hypervectors, and approximately quadratically for sparse binary hypervectors.

In summary, we introduced an attractor-based neural state machine which overcomes many of the shortcomings that made previous models unsuitable for use in biology, and propose that attractor-based FSMs represent a plausible path by which FSMs may exist as a distributed computational primitive in biological neural networks.

## REFERENCES

Alibart, F., Zamanidoost, E., & Strukov, D. B. (2013). Pattern classification by memristive crossbar circuits using ex situ and in situ training. _Nature Communications_, _4_(1), 2072. 10.1038/ncomms3072.

Alquézar, R., & Sanfeliu, A. (1995). An algebraic framework to represent finite state machines in single-layer recurrent neural networks. _Neural Computation_, _7_. 10.1162/neco.1995.7.5.931.

Amari, S.-i. (1989). Characteristics of sparsely encoded associative memory. _Neural Networks_, 2(6), 451-457. 10.1016/0893-6080(89)90043-9.

Amit, D. J. (1988). Neural networks counting chimies. _Proceedings of the National Academy of Sciences of the United States of America_, _85_(7), 2141-2145. 10.1073/pnas.85.7.2141.

Amit, D. J. (1989). Modeling brain function: The world of attractor neural networks, 1st edition. 10.1017/CBO9780511623257.

Amit, D. J., & Fusi, S. (1994). Learning in neural networks with material synapses. _Neural Computation_, _6_(5), 957--982. 10.1162/neco.1994.6.5.957.

Auge, D., Hille, J., Mueller, E., & Knoll, A. (2021). A survey of encoding techniques for signal processing in spiking neural networks. _Neural Processing Letters_, _53_(6), 4693--4710. 10.1007/s11063-021-10562-2.

Backus, J. (1978). Can programming be liberated from the von Neumann style? A functional style and its algebra of programs. _Communications of the ACM_, _21_(8), 613- 641. 10.1145/359576.359579.

Baldassi, C., Gerace, F., Lucibello, C., Saglietti, L., & Zecchina, R. (2016). Learning may need only a few bits of synaptic precision. _Physical Review E_, _93_. 10.1103/PhysRevE.93.052313.

Barth, A. L., & Poulet, J. F. A. (2012). Experimental evidence for sparse firing in the neocortex. _Trends in Neuroscience_, _35_(6), 345--355. 10.1016/j.tins.2012.03.008.

Bartol, T. M., Jr, Bromer, C., Kinney, J., Chirillo, M. A., Bourne, J. N., Harris, K. M., & Sejnowski, T. J. (2015). Nanoconnectonic upper bound on the variability of synaptic plasticity. _eLife_, _4_, e10778. 10.7554/eLife.10778.

Brinkman, B. A. W., Yan, H., Maffei, A., Park, I. M., Fontanini, A., Wang, J., & La Camera, G. (2022). Metastable dynamics of neural circuits and networks. _Applied Physics Reviews_, _9_(1), 011313. 10.1063/5.0062603.

Brunel, N., Carusi, F., & Fusi, S. (1998). Slow stochastic Hebbian learning of classes of stimuli in a recurrent neural network. _Network: Computation in Neural Systems_, _9_(1), 123-152. 10.1088/0954-898X\(\cdot\)9 \(\cdot\)1 O07.

Buhmann, J., & Schulten, K. (1987). Noise-driven temporal association in neural networks. _Europhysics Letters (EPL)_, _4_(10), 1205--1209. 10.1209/0295- 5075/4/10/021.

Buonomano, D. V., & Maass, W. (2009). State-dependent computations: Spatiotemporal processing in cortical networks. _Nature Reviews Neuroscience_, _10_(2), 113- 125. 10.1038/nrn2558.

Chaudhuri, R., & Fiete, I. (2016). Computational principles of memory. _Nature Neuroscience_, _19_(3), 394-403. 10.1038/nn.4237.

Chen, B., & Miller, P. (2020). Attractor-state itinerancy in neural circuits with synaptic depression. _The Journal of Mathematical Neuroscience_, _10_(1), 15. 10.1186/s13408-020-00093-w.

Clarkson, K. L., Ubaru, S., & Yang, E. (2023). Capacity analysis of vector symbolic architectures. 10.48550/arXiv.2301.10352.

Crawford, E., Gingerich, M., & Eliasmith, C. (2016). Biologically plausible, human-scale knowledge representation. _Cognitive Science_, _40_(4), 782-821. 10.1111/cogs.12261.

Daelli, V., & Treves, A. (2010). Neural attractor dynamics in object recognition. _Experimental Brain Research_, _203_(2), 241--248. 10.1007/s00221-010-2243-1.

Das, S., & Mozer, M. C. (1994). A unified gradient- descent/clustering architecture for finite state machine

16

---

## Page 17

induction. _Advances in Neural Information Processing Systems, 6_.<tbc>

Davies, M., Srinivasa, N., Lin, T.-H., Chinya, G., Cao, Y., Choday, S. H., Dimou, G., Joshi, P., Imam, N., Jain, S., Liao, Y., Lin, C.-K., Lines, A., Liu, R., Mathaikutty, D., McCoy, S., Paul, A., Tse, J., Venkataramanan, G., ... Wang, H. (2018). Loihi: A neuromorphic many- core processor with on-chip learning. _IEEE Micro_, 38(1), 82-99. 10.1109/MM.2018.112130359.<tbc>

Dayan, P. (2008). Simple substrates for complex cognition. _Frontiers in Neuroscience_, 2, 31. 10.3389/neuro.01.031.2008.<tbc>

Drossaers, M. F. J. (1992). Hopfield models as nondeterministic finite-state machines. _Proceedings of the 14th conference on Computational linguistics - Volume 1_, 113-119. 10.3115/992066.992087.<tbc>

Eliasmith, C. (2005). A unified approach to building and controlling spiking attractor networks. _Neural Computation_, _17_(6), 1276-1314. 10.1162/0899766053630332.<tbc>

Forcada, M., & Carrasco, R. C. (2001). Finite-state computation in analog neural networks: Steps towards biologically plausible models? _Emergent Neural Computational Architectures Based on Neuroscience_. 10.1007/3-540-44597-8 ̇34.<tbc>

Frady, E. P., & Sommer, F. T. (2019). Robust computation with rhythmic spike patterns. _Proceedings of the National Academy of Sciences_, _116_(36), 18050-18059. 10.1073/pnas.1902653116.<tbc>

Furber, S. B., Bainbridge, W. J., Cumpstey, J. M., & Temple, S. (2004). Sparse distributed memory using N-of-M codes. _Neural Networks: The Official Journal of the International Neural Network Society_, _17_(10), 1437- 1451. 10.1016/j.neunet.2004.07.003.<tbc>

Gayler, R. W. (1998). Multiplicative binding, representation operators & analogy. _Advances in Analogy Research: Integration of Theory and Data from the Cognitive, Computational, and Neural Sciences_.<tbc>

Granger, R. (2020). Toward the quantification of cognition. _arXiv:2008.05580 [cs, q-bio]_. 10.48550/arXiv.2008.05580.<tbc>

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing machines. 10.48550/arXiv.1410.5401.<tbc>

Grefenstette, E., Hermann, K. M., Suleyman, M., & Blunsom, P. (2015). Learning to transduce with unbounded memory. _Advances in Neural Information Processing Systems_, _28_. 10.48550/arXiv.1506.02516.<tbc>

Gritsenko, V. I., Rachkovskij, D. A., Frolov, A. A., Gayler, R., Kleyko, D., & Osipov, E. (2017). Neural distributed autoassociative memories: A survey. _Kibernetika i vyˇcislitel’naå tehnika_, _2017_(2(188)), 5-35. 10.15407/kvt188.02.005.<tbc>

Groschner, L. N., Malis, J. G., Zuidinga, B., & Borst, A. (2022). A biophysical account of multiplication by a single neuron. _Nature_, _603_(7899), 119-123. 10.1038/s41586-022-04428-3.<tbc>

Gutfreund, H., & Mezard, M. (1988). Processing of temporal sequences in neural networks. _Physical Review Letters_, _61_(2), 235-238. 10.1103/PhysRevLett.61.235.<tbc>

Hebb, D. O. (1949). _The organization of behavior; a neuropsychological theory_.<tbc>

Hersche, M., Zeqiri, M., Benini, L., Sebastian, A., & Rahimi, A. (2023). A neuro-vector-symbolic architecture for solving Raven’s progressive matrices. _Nature Machine Intelligence_, _5_(4), 363-375. 10.1038/s42256- 023-00630-8.<tbc>

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. _Proceedings of the National Academy of Sciences_, _79_(8), 2554-2558. 10.1073/pnas.79.8.2554.<tbc>

Ielmini, D., & Wong, H.-S. P. (2018). In-memory computing with resistive switching devices. _Nature Electronics_, _1_(6), 333-343. 10.1038/s41928-018-0092-2.<tbc>

Indiveri, G., & Liu, S.-C. (2015). Memory and information processing in neuromorphic systems. _Proceedings of the IEEE_, _103_(8), 1379-1397. 10.1109/JPROC.2015.2444094.<tbc>

Kanerva, P. (1997). Fully distributed representation. _Proc. 1997 Real World Computing Symposium (RWC97, Tokyo)_, 358-365.<tbc>

Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. _Cognitive Computation_, _1_(2), 139-159. 10.1007/s12559-009- 9009-8.<tbc>

Khacef, L., Klein, P., Cartiglia, M., Rubino, A., Indiveri, G., & Chicca, E. (2022). Spike-based local synaptic plasticity: A survey of computational models and neuromorphic circuits. 10.48550/arXiv.2209.15536.<tbc>

Khona, M., & Fiete, I. R. (2022). Attractor and integrator networks in the brain. _Nature Reviews Neuroscience_, _23_(12), 744-766. 10.1038/s41583-022-00642-0.<tbc>

Kleinfeld, D. (1986). Sequential state generation by model neural networks. _Proceedings of the National Academy of Sciences of the United States of America_, _83_(24), 9469-9473. 10.1073/pnas.83.24.9469.<tbc>

Kleyko, D., Davies, M., Frady, E. P., Kanerva, P., Kent, S. J., Olshausen, B. A., Osipov, E., Rabaey, J. M., Rachkovskij, D. A., Rahimi, A., & Sommer, F. T. (2021). Vector symbolic architectures as a computing framework for nanoscale hardware. _arXiv:2106.05268 [cs]_. 10.48550/arXiv.2106.05268.<tbc>

Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2022). A survey on hyperdimensional computing aka vector symbolic architectures, Part I: Models and data transformations. _ACM Computing Surveys_. 10.1145/3538531.<tbc>

Koch, C. (1998). _Biophysics of computation: Information processing in single neurons_. 10.1093/oso/9780195104912.001.0001.<tbc>

17

---

## Page 18

Krotov, D., & Hopfield, J. (2021). Large associative memory problem in neurobiology and machine learning. _International Conference on Learning Representations_.<tbc>

Lee Giles, C., Horne, B. G., & Lin, T. (1995). Learning a class of large finite state machines with a recurrent neural network. _Neural Networks_, _8_(9), 1359-1365. 10.1016/0893-6080(95)00041-0.<tbc>

Li, Y., Xiao, T. P., Bennett, C. H., Isele, E., Melanianas, A., Tao, H., Marinella, M. J., Salleo, A., Fuller, E. J., & Talin, A. A. (2021). In situ parallel training of analog neural network using electrochemical random-access memory. _Frontiers in Neuroscience_, _15_, 636127. 10.3389/fnins.2021.636127.<tbc>

Liang, D., Kreiser, R., Nielsen, C., Qiao, N., Sandamirskaya, Y., & Indiveri, G. (2019). Neural state machines for robust learning and control of neuromorphic agents. _IEEE Journal on Emerging and Selected Topics in Circuits and Systems_, _9_(4), 679-689. 10.1109/JET- CAS.2019.2951442.<tbc>

Lin, A. C., Bygrave, A. M., de Calignon, A., Lee, T., & Miesenböck, G. (2014). Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination. _Nature Neuroscience_, _17_(4), 559-568. 10.1038/nn.3660.<tbc>

Little, W. A. (1974). The existence of persistent states in the brain. _Mathematical Biosciences_, _19_(1), 101-120. 10.1016/0025-5564(74)90031-5.<tbc>

Liu, S.-C., Delbruck, T., Indiveri, G., Whatley, A., & Douglas, R. (2014). _Event-based neuromorphic systems_. 10.1002/9781118927601.<tbc>

Mali, A. A., Ororbia II, A. G., & Giles, C. L. (2020). A neural state pushdown automata. _IEEE Transactions on Artificial Intelligence_, _1_(3), 193-205. 10.1109/TAI.2021.3055167.<tbc>

Manohar, R. (2022). Hardware/software co-design for neuromorphic systems. _2022 IEEE Custom Integrated Circuits Conference (CICC)_, 01-05. 10.1109/CICCC53496.2022.9772863.<tbc>

Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. _Nature_, _503_(7474), 78- 84. 10.1038/nature12742.<tbc>

Miller, P. (2016). Itinerancy between attractor states in neural systems. _Current opinion in neurobiology_, _40_, 14-22. 10.1016/j.cond.2016.05.005.<tbc>

Minsky, M. L. (1967). _Computation: Finite and infinite machines_.<tbc>

Neftci, E., Binas, J., Rutishauser, U., Chicca, E., Indiveri, G., & Douglas, R. J. (2013). Synthesizing cognition in neuromorphic electronic systems. _Proceedings of the National Academy of Sciences_, _110_(37), E3468- E3476. 10.1073/pnas.1212083110.<tbc>

Nielsen, C., Qiao, N., & Indiveri, G. (2017). A compact ultra low-power pulse delay and extension circuit for neuromorphic processors. _2017 IEEE Biomedi_<tbc>

_cal Circuits and Systems Conference (BioCAS)_, 1-4. 10.1109/BIOCAS.2017.8325234.<tbc>

Noest, A. J. (1987). Phasor neural networks. _Proceedings of the 1987 International Conference on Neural Information Processing Systems_, 584-591.<tbc>

O’Connor, D. H., Wittenberg, G. M., & Wang, S. S.-H. (2005). Graded bidirectional synaptic plasticity is composed of switch-like unitary events. _Proceedings of the National Academy of Sciences of the United States of America_, _102_(27), 9679-9684. 10.1073/pnas.0502332102.<tbc>

Olshausen, B. A., & Field, D. J. (2004). Sparse coding of sensory inputs. _Current Opinion in Neurobiology_, _14_(4), 481-487. 10.1016/j.cond.2004.07.007.<tbc>

Omlin, C., Thornber, K., & Giles, C. (1998). Fuzzy finite-state automata can be deterministically encoded into recurrent neural networks. _IEEE Transactions on Fuzzy Systems_, _6_(1), 76-89. 10.1109/91.660809.<tbc>

Orchard, J., & Jarvis, R. (2023). Hyperdimensional computing with spiking-phasor neurons. _Proceedings of the 2023 International Conference on Neuromorphic Systems_, 1-7. 10.1145/3589737.3605982.<tbc>

Osipov, E., Kleyko, D., & Legalov, A. (2017). Associative synthesis of finite state automata model of a controlled object with hyperdimensional computing. _IECON 2017 - 43rd Annual Conference of the IEEE Industrial Electronics Society_, 3276-3281. 10.1109/IECON.2017.8216554.<tbc>

Plate, T. A. (1995). Holographic reduced representations. _IEEE Transactions on Neural Networks_, _6_(3), 623-641. 10.1109/72.377968.<tbc>

Plate, T. A. (2003). _Holographic reduced representation: Distributed representation for cognitive structures_.<tbc>

Poduval, P., Alimohamadi, H., Zakeri, A., Imani, F., Najafi, M. H., Givargis, T., & Imani, M. (2022). GraphD: Graph-based hyperdimensional memorization for brain-like cognitive learning. _Frontiers in Neuroscience_, _16_. 10.3389/fnins.2022.757125.<tbc>

Pollack, J. B. (1991). The induction of dynamical recognizers. _Machine Learning_, _7_(2), 227-252. 10.1007/BF00114845.<tbc>

Recanatesi, S., Katkov, M., & Tsodyks, M. (2017). Memory states and transitions between them in attractor neural networks. _Neural Computation_, _29_(10), 2684-2711. 10.3389/fncom.2010.00024.<tbc>

Rigotti, M., Ben Dayan Rubin, D., Wang, X.-J., & Fusi, S. (2010). Internal representation of task rules by recurrent dynamics: The importance of the diversity of neural responses. _Frontiers in Computational Neuroscience_, _4_. 10.3389/fncom.2010.00024.<tbc>

Rolls, E. (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. _Frontiers in systems neuroscience_, _7_, 74. 10.3389/fn- sys.2013.00074.<tbc>

18

---

## Page 19

Rolls, E. T., & Treves, A. (2011). The neuronal encoding of information in the brain. _Progress in Neurobiology_, 95(3), 448-490. 10.1016/j.pneurobio.2011.08.002.<tbc>

Rumelhart, D. E., & McClelland, J. L. (1986). Parallel distributed processing: Explorations in the microstructure of cognition. Volume 1. Foundations.<tbc>

Rutishauser, U., & Douglas, R. (2009). State-dependent computation using coupled recurrent networks. _Neural computation_, _21_, 478-509. 10.1162/neco.2008.03-08- 734.<tbc>

Schlag, I., Smolensky, P., Fernandez, R., jojic, N., Schmidhuber, J., & Gao, J. (2020). Enhancing the transformer with explicit relational encoding for math problem solving. 10.48550/arXiv.1910.06611.<tbc>

Schneidman, E., Berry, M. J., Segev, R., & Bialek, W. (2006). Weak pairwise correlations imply strongly correlated network states in a neural population. _Nature_, _440_(7087), 1007-1012. 10.1038/nature04701.<tbc>

Sompolinsky, H. (1987). The theory of neural networks: The Hebb rule and beyond. In J. L. van Hemmen & I. Morgenstern (Eds.), _Heidelberg Colloquium on Glassy Dynamics_ (pp. 485-527). 10.1007/BFb0057531.<tbc>

Sompolinsky, H., & Karter, I. (1986). Temporal association in asymmetric neural networks. _Physical Review Letters_, _57_(22), 2861-2864. 10.1103/PhysRevLett.57.2861.<tbc>

Steinberg, J., & Sompolinsky, H. (2022). Associative memory of structured knowledge. _Scientific Reports_, _12_(1), 21808. 10.1038/s41598-022-25708-y.<tbc>

Tajima, S., Koida, K., Tajima, C. I., Suzuki, H., Aihara, K., & Komatsu, H. (2017). Task-dependent recurrent dynamics in visual cortex. _eLife_, _6_, e26868. 10.7554/eLife.26868.<tbc>

Teeters, J. L., Kleyko, D., Kanerva, P., & Olshausen, B. A. (2023). On separating long- and short-term memories in hyperdimensional computing. _Frontiers in Neuroscience_, _16_. 10.3389/fnins.2022.867568.<tbc>

Tsodyks, M. V., & Feigel’man, M. V. (1988). The enhanced storage capacity in neural networks with low activity level. _Europhysics Letters (EPL)_, _6_(2), 101-105. 10.1209/0295-5075/6/2/002.<tbc>

Verleysen, M., & Jespers, P. (1989). An analog VLSI implementation of Hopfield’s neural network. _IEEE Micro_, _9_(6), 46-55. 10.1109/40.42986.<tbc>

Xia, Q., & Yang, J. J. (2019). Memristive crossbar arrays for brain-inspired computing. _Nature Materials_, _18_(4), 309-323. 10.1038/s41563-019-0291-x.<tbc>

Yerxa, T., Anderson, A., & Weiss, E. (2018). The hyperdimensional stack machine. _Cognitive Computing_, 1-2.<tbc>

Zeng, Z., Goodman, R. M., & Smyth, P. (1993). Learning finite state machines with self-clustering recurrent networks. _Neural Computation_, _5_(6), 976-990. 10.1162/neco.1993.5.6.976.<tbc>

Zenke, F., & Neftci, E. O. (2021). Brain-inspired learning on neuromorphic substrates. _Proceedings of the IEEE_, _109_(5), 935-950. 10.1109/JPROC.2020.3045625.<tbc>

Zidan, M. A., & Lu, W. D. (2020). Chapter 9 - vector multiplications using memristive devices and applications thereof. In S. Spiga, A. Sebastian, D. Querlioz, & B. Rajendran (Eds.), _Memristive Devices for Brain- Inspired Computing_ (pp. 221-254). 10.1016/B978-0- 08-102782-0.00009-5.<tbc>

19

---

## Page 20

APPENDIX

### A. Dynamics without masking

For the following calculations we assume that the coding level of the output states _f<sub>r</sub>_ is low enough that their effect can be ignored. With this in mind, if we ignore the semantic differences between attractors for node states and attractors for edge states, the two summations over states can be absorbed into one summation over both types of attractor, here both denoted \(\mathbf{x}^\nu\). Similarly there is then no difference between the two transition cross-terms within each **E** term, and they too can be absorbed into one summation. Our simplified expression for **W** is now given by

\(\mathbf{W}=\frac{1}{N}\sum_{\text{attr}^\text{s}\nu}^{N_Z+N_E}\mathbf{x}^\nu\mathbf{x}^{\nu\mathsf{T}}+\frac{1}{N}\sum_{\text{ran}^\text{s}\lambda}^{2N_E}H(\mathbf{s}^{\pi(\lambda)})\circ(\mathbf{x}^{\upsilon(\lambda)}-\mathbf{x}^{\chi(\lambda)})(\mathbf{x}^{\chi(\lambda)}\circ\mathbf{s}^{\pi(\lambda)})^\mathsf{T}\) (35)

where \(\chi(\lambda)\) and \(\upsilon(\lambda)\) are functions \(\{1,\dots,2N_E\}\rightarrow\{1,\dots,N_Z+N_E\}\) determining the indices of the source and target states for transition \(\lambda\), and \(\pi(\lambda):\{1,\dots,2N_E\}\rightarrow\{1,\dotsN_\text{stimuli}\}\) determines the index of the associated stimulus. We then wish to calculate the statistics of the postsynaptic sum **Wz** while the attractor network is currently in an attractor state. When in an attractor state \(\mathbf{x}^\mu\), the postsynaptic sum is given by

\(\big[\mathbf{Wx}^\mu\big]_i=\frac{1}{N}\sum_{\text{attr}^\text{s}\nu}^{N_Z+N_E}x_i^\nu\underbrace{\big[\mathbf{x}^\nu\cdot\mathbf{x}^\mu\big]_{N\text{if}\mu=\nu}}_{\text{else}\mathcal{N}(0,N)}+\frac{1}{N}\sum_{\text{ran}^\text{s}\lambda}^{2N_E}H(s_i^{\pi(\lambda)})\circ(x_i^{\upsilon(\lambda)}-x_i^{\chi(\lambda)})\underbrace{\big[(\mathbf{x}^{\chi(\lambda)}\circ\mathbf{s}^{\pi(\lambda)})\cdot\mathbf{x}^\mu\big]}_{\mathcal{N}(0,N)}=x_i^\mu+\sum_{\text{attr}^\text{s}\nu\neq\mu}^{N_Z+N_E}\underbrace{x_i^\nu_{\text{Var}.=1}}\big[\mathcal{N}^\nu\big(0,\frac{1}{N}\big)\big]+\sum_{\text{ran}^\text{s}\lambda}^{2N_E}\underbrace{H(s_i^{\pi(\lambda)})\circ(x_i^{\upsilon(\lambda)}-x_i^{\chi(\lambda)})}_{\text{Var}.=1}\big[\mathcal{N}^\lambda\big(0,\frac{1}{N}\big)\big]\approx x_i^\mu+\mathcal{N}\big(0,\frac{N_Z+N_E-1}{N}\big)+\mathcal{N}\big(0,\frac{2N_E}{N}\big)\approx x_i^\mu+\mathcal{N}\big(0,\frac{N_Z+3N_E}{N}\big)\text{(36)}\)

where we have used the notation \(\mathcal{N}(\mu,\sigma^2)\) to denote a normally-distributed random variable (RV) with mean \(\mu\) and variance \(\sigma^2\). In the third line we have made the approximation in the transition summation that the linear sum of attractor hypervectors, each multiplied by a Gaussian RV, is itself a separate Gaussian RV in each dimension. This holds as long as there are "many" attractor terms appearing on the LHS of the transition summation. Said otherwise, if the summation over transition terms has only very few unique attractor terms on the LHS (\(N_E\gg N_Z\)), then the noise will be a random linear sum of the same few (masked) hypervectors, each with approximate magnitude \(\frac{1}{\sqrt{N}}\), and so will be highly correlated between dimensions. Nonetheless we assume we are far away from this regime, and let the effect of the sum of these unwanted terms be approximated by a normally-distributed random vector, and so we have

\(\mathbf{Wx}^\mu\approx\mathbf{x}^\mu+\sigma\mathbf{n}\) (37)

where \(\sigma=\sqrt{\frac{N_Z+3N_E}{N}}\) is the strength of cross-talk noise, and **n** a vector composed of IID standard normally-distributed RVs. This procedure of quantifying the signal-to-noise ratio (SNR) is adapted from that in the original Hopfield paper (Amit 1989; Hopfield 1982).

### B. Dynamics with masking

We can similarly calculate the postsynaptic sum when in an attractor state \(\mathbf{x}^\mu\), while the network is being masked by a stimulus \(\mathbf{s}^\kappa\), with this (state, stimulus) tuple corresponding to a certain valid transition \(\lambda'\), with source, target, and stimulus<tbc>

20

---

## Page 21

hypervectors \(\mathbf{x}^\mu\), \(\mathbf{x}^\phi\), and \(\mathbf{s}^\kappa\) respectively:<tbc>

\(\big[\mathbf{W}\big(\mathbf{x}^\mu\circ H(\mathbf{s}^\kappa)\big)\big]_i=\frac{1}{N}\sum_{\text{attr}^\text{s}\nu}^{N_Z+N_E}x_i^\nu\underbrace{\big[\mathbf{x}^\nu\cdot\big(\mathbf{x}^\mu\circ H(\mathbf{s}^\kappa)\big)\big]}_\underline{\frac{1}{\underline{1}}\underline{N}\text{if}\mu=\nu}\mathtt{else}\mathcal{N}(0,\frac{1}{\underline{2}}N)}+\frac{1}{N}\sum_{\text{ran}^\text{s}\lambda}^{2N_E}H(s_i^{\pi(\lambda)})(x_i^{\nu(\lambda)}-x_i^{\chi(\lambda)})\underbrace{\left[(\mathbf{x}^{\chi(\lambda)}\circ\mathbf{s}^{\pi(\lambda)})\cdot\big(\mathbf{x}^\mu\circ H(\mathbf{s}^\kappa)\big)\right]}_\underline{\frac{1}{\underline{2}}\text{if}\chi(\lambda)=\mu\text{and}\pi(\lambda)=\kappa}==\frac{1}{2}x_i^\mu+\frac{1}{2}H(s_i^\kappa)(x_i^\phi-x_i^\mu)+\sum_{\text{attr}^\text{s}\nu\neq\mu}^{N_Z+N_E}\underbrace{x_i^\nu}_\mathtt{Var}.=1\big[\mathcal{N}^\nu(0,\frac{1}{2N})\big]+\sum_{\text{ran}^\text{s}\lambda\neq\lambda^\prime}^{2N_E}\underbrace{H(s_i^{\pi(\lambda)})(x_i^{\nu(\lambda)}-x_i^{\chi(\lambda)})}_\mathtt{Var}.=1\big(\mathcal{N}^\lambda(0,\frac{1}{2N})\big)\approx\frac{1}{2}\big[H(s_i^\kappa)+H(-s_i^\kappa)\big]x_i^\mu+\frac{1}{2}H(s_i^\kappa)(x_i^\phi-x_i^\mu)+\mathcal{N}\big(0,\frac{N_Z+N_E-1}{2N}\big)+\mathcal{N}\big(0,\frac{2N_E-1}{2N}\big)=\frac{1}{2}H(s_i^\kappa)x_i^\phi+\frac{1}{2}H(-s_i^\kappa)x_i^\mu+\mathcal{N}\big(0,\frac{N_Z+3N_E-2}{2N}\big)\approx\frac{1}{2}\big[H(s_i^\kappa)x_i^\phi+H(-s_i^\kappa)x_i^\mu+\sqrt{2}\cdot\mathcal{N}\big(0,\frac{N_Z+3N_E}{N}\big)\big]\text{(38)}\)

<tbc>where in the third line we have made the same approximations as previously discussed. The postsynaptic sum is thus approximately \(\mathbf{x}^\phi\) in all indices that are not currently being masked, which drives the network towards that (target) attractor. In vector form, the above is written as

\(\mathbf{W}\big(\mathbf{x}^\mu\circ H(\mathbf{s}^\kappa)\big)\unknown\unknown H(\mathbf{s}^\kappa)\circ\mathbf{x}^\phi+H(-\mathbf{s}^\kappa)\circ\mathbf{x}^\mu+\sqrt{2}\sigma\mathbf{n}\) (39)

where it is assumed that there exists a stored transition from state \(\mathbf{x}^\mu\) to \(\mathbf{x}^\phi\) with stimulus \(\mathbf{s}^\kappa\), and \(\unknown\) denotes approximate proportionality. A similar calculation can be performed in the case that a stimulus is imposed which does not correspond to a valid transition for the current state. In this case, no terms of significant magnitude emerge from the transition summation, and we are left with

\(\mathbf{W}\big(\mathbf{x}^\mu\circ H(\mathbf{s}^\text{invalid})\big)\unknown\unknown\mathbf{x}^\mu+\sqrt{2}\sigma\mathbf{n}\) (40)

i.e. the attractor dynamics are largely unaffected. Since we have not distinguished between our above attractor terms being node attractors or edge attractors, or our stimuli from being **s**_<sub>a</sub>_ or **s**_<sub>b</sub>_ stimuli, the above results can be applied to all relevant situations _mutatis mutandis_.

## C. Why model input as masking?

One immediate question might be why we have chosen to model input to the network as a masking of the neural state vector (Equation 14), rather than simply modelling input as a Hadamard product, with a state update rule given by<tbc>

\(\mathbf{z}_{t+1}=sgn\big(\mathbf{W}\big(\mathbf{z}_t\circ\mathbf{s}\big)\big)\) (41)

<tbc>such that a component for which the input stimulus _s<sub>i</sub>_=-1 triggers a "flip" in the neuron state \(+1\leftrightarrow-1\). As will be shown, the problem with this construction is that it relies on the synchrony of input to the network, and does not allow for for the input to arrive asynchronously and with arbitrary delays. While this would not be a problem for a digital synchronous system, such timing constraints cannot be expected to be met in a network of asynchronously-firing biological neurons. In the synchronous case however, the edge terms \(\mathbf{E}^\eta\) in the weights matrix construction could be simplified to<tbc>

\(\mathbf{E}^{(\eta)}=\mathbf{y}\big(\mathbf{x}\circ\mathbf{s}\big)^\mathsf{T}\) (42)

<tbc>where as per previous notation, **x** and **y** are the source and target attractor states respectively, and **s** the stimulus to cause the transition. Superficially, this construction would then satisfy our main requirements for achieving the desired attractor itinerancy dynamics during input and rest scenarios, namely<tbc>

\(\mathbf{Wx}\approx\mathbf{x}\forall\mathbf{x}\in X_\mathrm{AN}\) (43)

21

---

## Page 22

![picture 1](figures/page_0022_fig_01.png)
*Fig. 9: An attractor network constructed via the simpler weights construction method specified in Section VII-C, with input to the network modelled as Hadamard product binding, rather than component-wise masking. **a)** The similarity of the network state **z**<sub>_t_</sub> to stored node hypervectors, when the stimulus hypervector **s** is applied on one time step for all neurons _simultaneously_. **b)** A subset of the stimulus hypervector **s** at each time step in this _synchronous_ case. **c)** The attractor overlaps in the _asynchronous_ case, where the stimulus **s** is applied over multiple time steps randomly. **d)** A subset of the stimulus hypervector **s** at each time step in this _asynchronous_ case. For visual clarity, the two stimulus hypervectors shown were manually chosen rather than randomly generated. In the synchronous case, the network performs the correct walk between attractor states as intended. In the asynchronous case however, the stimuli fail to effect the desired transitions, since any changes in the network state caused by the input stimuli are short-lived, as they are quickly reversed on the next time step by the attractor network’s pattern-correcting dynamics.*

which ensures that while there is no input to the network, the states **x** are stable attractors of the network dynamics, and

\(\mathbf{W}\big(\mathbf{x}\circ\mathbf{s}\big)\approx\mathbf{y}\) (44)

which ensures that inputting the stimulus **s** triggers the desired transition. The resulting dynamics for this network - when input is entirely synchronous - are shown in Figure 9a , and indeed the network performs the desired walk.

We then test the functionality of the attractor network with Hadamard input when the exact simultaneous arrival of input stimuli cannot be guaranteed, i.e. the input to the network is asynchronous. To model this, we consider that the arrival time of the stimulus is component-wise randomly and uniformly spread over 5 time steps, rather than just one. The same attractor network receiving the same sequence of Hadamard-product stimuli, but now asynchronously, is shown in Figure 9 c). The network does not perform the correct walk between attractor states, and instead remains localised near the initial attractor state across all time steps. This is due to the fact that, although when input is applied, the network begins to move away from the initial attractor state, these changes are immediately undone by the network’s inherent attractor dynamics, since the neural state is still within the initial attractor’s basin of attraction. Only when the timescale of the input is far faster than the timescale of the attractor dynamics (e.g. input is synchronous) may the input accumulate fast enough to escape the initial basin of attraction.

When input to the network is treated as masking operation however (Equation 14), the attractor itinerancy dynamics are robust to input asynchrony. To model this, the input stimulus is stochastically applied, with each component being delayed

22

---

## Page 23

![picture 1](figures/page_0023_fig_01.png)
*Fig. 10: The attractor network performing a walk as masking input is applied asynchronously over multiple time steps with random delays. **a)** The similarities between the network state **z**_<sub>t</sub>_ and stored node hypervectors \(\mathbf{x}\in X_\mathrm{AN}\). **b)** A subset of the stimulus hypervector **s** being applied to the network as a mask at each time step. Indices which are black on any time step have \[**s**\]_<sub>i</sub>_=-1 and so are being masked by the stimulus. For visual clarity, the two stimulus hypervectors shown were manually chosen, rather than randomly generated. The attractor transition dynamics are thus robust to input asynchrony when the input is modelled as a component-wise masking of the network state.*

<tbc>randomly and uniformly by up to 20 time steps. The stimulus is then held for 10 time steps, and stochastically removed over 20 time steps in the same manner. The attractor network with asynchronous masking input is shown in Figure 10, and functions as desired, performing the correct walk between attractor states. Modelling input to the network as a masking operation thus allows the network to operate robustly in asynchronous regimes, while modelling input to the network as a Hadamard product does not.

## D. The need for edge states

The need for the edge state attractors arises when one wants to emulate an FSM where there are consecutive edges with the same stimulus. For example, in the FSM implemented throughout this article (Figure 1) there is an incoming edge from "Zeus" to "Kronos" with stimulus "father\_is" and then immediately an outgoing edge from "Kronos" to "Uranus" with stimulus "father\_is" also. More generally, consider that we wish to embed the transitions

\(\mathbf{x}_1\longrightarrow^\mathbf{s}\mathbf{x}_2\longrightarrow^\mathbf{s}\mathbf{x}_3\) (45)

In the fully synchronous case, i.e. when input is applied for one time step only, there is no need for edge states. When the stimulus **s** is applied, the network will make one transition only. In the asynchronous case however, one cannot ensure that the stimulus is applied for one time step only. Thus, starting from **x**<sub>1</sub>, when the stimulus is applied "once" for an arbitrary number of time steps, the network may have the unwanted behaviour of transitioning to **x**<sub>2</sub> on the first time step, and then to **x**<sub>3</sub> on the second, effectively overshooting and skipping **x**<sub>2</sub>. In Figure 11 we see the dynamics of the attractor network constructed without any edge states, with inputs which are applied for 10 time steps each, and we indeed see the undesirable skipping behaviour. Similarly, bidirectional edges with the same stimulus (e.g. “consort\_is”) cause an unwanted oscillation between attractor states. The edge states offer a solution to this problem: by adding an intermediate attractor state for every edge, and splitting each edge into two transitions with stimuli **s**_<sub>a</sub>_ and **s**_<sub>b</sub>_, we ensure that there are no consecutive edges with the same stimulus.

If we don’t necessarily need to be able to embed FSMs with consecutive edges with the same stimulus, then we can rid of the edge states, and construct our weights matrix with simpler transition terms like in Equation 34. An attractor network constructed in this way is shown in Figure 12, for a chosen FSM that does not require edge states, but still contains state- dependent transitions. The network performs the correct walk between attractor states as intended, and does not suffer from any of the unwanted skipping or oscillatory phenomena like in Figure 11. Thus, while the edge states are required to ensure that any FSM can be implemented in a "large enough" attractor network, they are not strictly necessary to achieve state-dependent stimulus-triggered attractor transition dynamics.

## E. Sparse stimuli

One shortcoming of the model might be that we used dense bipolar hypervectors **s** to represent the stimuli, meaning that when **s** is being input to the network, masking all neurons for which _s<sub>j</sub>_=-1, approximately half of all neurons within<tbc>

23

---

## Page 24

![picture 1](figures/page_0024_fig_01.png)
*Fig. 11: An attractor network receiving a sequence of stimuli to trigger a certain walk constructed **a**) without edge states and **b**) with edge states, with edge state overlaps being shown in **c**). Due to the consecutive edges in the FSM (Figure 1) with the same stimulus "father\_is", the edge-state-less network overshoots and skips the "Kronos" state, stopping instead at the "Uranus" state. Similarly, there is an unwanted oscillation between the states "Gaia" and "Uranus" due to the bidirectional edge with stimulus "consort\_is". The addition of the edge state attractors resolves these issues, and allows the network to function robustly when input stimuli are applied for an arbitrary number of time steps.*

<tbc>the network are silenced. This was initially chosen because unbiased bipolar hypervectors are arguably the simplest and most common choice of VSA representation, and highlights the fact that VSA-based methods can be applied to the design of attractor networks with very little required tweaking (Gayler 1998; Kleyko, Rachkovskij, et al. 2022).

From the biological perspective however, it could be seen as somewhat implausible that the number of active neurons should change so drastically (halving) while a stimulus is present. Furthermore, if implemented with spiking neurons, the large changes in the total spiking activity could cause unwanted effects in the spike rate of the non-masked neurons. Also, this means that while the network is being masked, the size of the network (and so its capacity) is reduced to _N/_2, and so the network is especially prone to instability during the transition periods, if the network is nearing its memory capacity limits.

For these reasons, it is worth exploring whether the network could be constructed such that during a masking operation, fewer than half of all neurons are masked, i.e. **s** is biased to contain more +1 than -1 entries<sup>3</sup>. To keep the notation consistent with the notation used for sparse binary hypervectors, we will denote the coding level of the attractor states as _f<sub>z</sub>_ (where previously it was simply _f_) and the coding level of the stimulus hypervectors as _f<sub>s</sub>_. The coding level of the stimulus hypervectors _f<sub>s</sub>_ we define to be the fraction of components for which _s<sub>j</sub>\>_0. A stimulus hypervector with _f<sub>s</sub>\>_0.5 thus silences fewer neurons from the network during a masking operation. This is not the only change we need to make however. If we turn to our (sparse) edge terms (Equation 27), they were previously constructed such that they would produce a non-negligible overlap with the network state **z**<sub>sp</sub> if and only if the network is in the correct attractor state _and_ is being masked by the correct stimulus. The important condition to be fulfilled is then

\(\mathbb{E}\left[\left[((\mathbf{x}_\mathrm{sp}-f\mathbf{1})\circ\mathbf{s})\cdot\mathbf{x}_\mathrm{sp}\right]_j\right]=^!0\forall j=1\dots N\) (46)

that is, the overlap should be negligible if the network is in the correct attractor state, but the stimulus is _not_ present. This condition is satisfied if the components of **s** are generated according to where _s<sub>j</sub>_ is the _j_’th component of **s**. This implies that for a stimulus hypervector biased towards having more positive entries (fewer neurons are masked), the negative entries must increase in magnitude to compensate for their infrequency. For the case that only a quarter of neurons are masked by the<tbc>

<sup>3</sup>We could also use binary **s** hypervectors, rather than positive/negative, and then alter the transition terms **E**_<sup>n</sup>_ to include _f_ and 1/(1-_f_) terms to achieve the same result. We believe it is more intuitive not to make this change for this section, however.

24

---

## Page 25

![picture 1](figures/page_0025_fig_01.png)

![picture 2](figures/page_0025_fig_02.png)
*Fig. 12: Embedding an FSM that does not require edge states, since it does not have consecutive edges with the same stimulus. **a)** The FSM to be embedded, representing a simple decision tree. **b) & c)** An attractor network constructed to store this FSM, without any edge states, as a sequence of stimuli is input. The network performs the correct walks between attractor states as desired. To note is that the second stimulus ("is\_orange") and its transition are state-dependent, as the target state ("carrot" or "tangerine") is dependent upon the stimulus given 20 time steps before ("is\_round" or "is\_pointy"). This illustrates that the edge states are not strictly necessary to implement state-dependent transitions between attractor states.*

<tbc>stimulus (_f<sub>s</sub>_=0.75), the negative 25% of components must have the value -3, while for _f<sub>s</sub>_=0.5 this of course collapses to the balanced bipolar hypervectors used throughout this article with **IP**(_S<sub>j</sub>_=1)=**IP**(_S<sub>j</sub>_=-1)=0.5 (Equation 2). We are forced to increase the magnitude of the negative terms, rather than reduce the positive terms, since the magnitude of the positive terms must remains identical to that of the stored attractor terms, in order to ensure that the correct target state is projected out during a transition. We can then construct our weights matrix in the same way as before, but using these biased stimulus hypervectors **s**. An attractor network was generated with coding levels _f<sub>z</sub>_=0.1 (10% of neurons are active in any attractor hypervector) and _f<sub>s</sub>_=0.9 (10% of neurons are masked by stimulus hypervectors), and the results are shown in Figure 13, with the neural state performing the correct walk between attractor states as desired.

To be noted is that as we approach \(f_s\rightarrow 1\), the stimuli become less and less distributed, with the limiting case _f<sub>s</sub>_=1-1_/N_ implying that only one component of **s** is negative, and so by masking only one neuron, the network will switch between attractor states. This case is obviously a stark departure from the robustness which the more distributed representations afford us, since if that single neuron is faulty or dies, it would be catastrophic for the functioning of the network. Similarly, if another independent stimulus were to, by chance, choose the same component to be non-negative, this would cause similarly unwanted<tbc>

25

\begin{tabular}{cc}
_s_ & **IP**(_s<sub>j</sub>=s_) \\
1 & _f_ \\
-_f/_(1-_f_) & (1-_f_) \\
\end{tabular}

---

## Page 26

![picture 1](figures/page_0026_fig_01.png)
*TABLE II: Notation and frequently used symbols.*

<tbc>dynamics. Less catastrophic, but still worth considering is that the noise added per edge term, as a result of the negative terms becoming very large, has variance that scales like Var\[_s<sub>j</sub>_\]≈1/(1-_f<sub>s</sub>_), and so for \(f_s\rightarrow 1\) contributes an increasing amount of unwanted noise to the system, destabilising the attractor dynamics. Nevertheless, this represents yet another trade-off in the attractor network’s design, as needing to mask fewer neurons might be worth the increased noise within the system, decreasing its memory capacity.

26

\begin{tabular}{cc}
Symbol & Definition \\
_N_ & Number of neurons within the attractor network \\
_N<sub>Z</sub>_ & Number of FSM states \\
_N<sub>E</sub>_ & Number of FSM edges \\
**a**,**b**,**c**_..._ & Dense bipolar hypervectors \\
**a**<sub>sp</sub>,**b**<sub>sp</sub>,**c**<sub>sp</sub>_..._ & Sparse binary hypervectors \\
_f_ & Coding level of a hypervector (fraction nonzero components) \\
**z**_<sub>t</sub>_ & Neuron state vector at time step _t_ \\
**x**,**y** & Node hypervectors representing an FSM state \\
**e** & Edge-state hypervectors \\
**s**,**s**_<sub>a</sub>_,**s**_<sub>b</sub>_ & Stimulus hypervectors \\
**r** & Ternary output hypervectors \\
**1** & A hypervector of all ones \\
**W** & Recurrent weights matrix \\
_w<sub>ij</sub>_ & Synaptic weight from neuron _j_ to _i_ \\
**E** & Matrices added to **W** to implement transitions \\
\(\circ\) & Hadamard product (component-wise multiplication) \\
**x**<sup>T</sup> & Transpose of **x** \\
\(H(\cdot)\) & Component-wise Heaviside function \\
\(\mathrm{sgn}(\cdot)\) & Component-wise sign function \\
\end{tabular}

---

