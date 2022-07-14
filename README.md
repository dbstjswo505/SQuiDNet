# SQuiDNet: Selective Query-guided Debiasing Network for Video Corpus Moment Retrieval

## Task: Video Corpus Moment Retrieval
Video moment retrieval (VMR) aims to localize target moments in untrimmed videos pertinent to given textual query. Existing retrieval systems tend to rely on retrieval bias as a shortcut and thus, fail to sufficiently learn multi-modal interactions between query and video.

SQuiDNet is proposed to debiasing in video moment retrieval via conjugating retrieval bias in either positive or negative way.

<p lign="center">
	<img src="./figs/Intro.PNG" alt="Intro" width="70%" height="70%"/>
</P>

## SQuiDNet Overview
SQuiDNet is composed of 3 modules: (a) BMR which reveals biased retrieval, (b) NMR which performs accurate retrieval, (c) SQuiD which removes bad biases from accurate retrieval of NMR subject to the meaning of query.

<p lign="center">
	<img src="./figs/Model.PNG" alt="Model" width="80%" height="80%"/>
</P>

## Implementation
### Setting
1. Clone the repositery

```
git clone Here.git
cd SQuiDNet
```

