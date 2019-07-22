# ANID 

Experiments for the paper:

**"A Neural Attention Model for Real-Time Network Intrusion Detection"** *2019 IEEE Conference on Local Computer Networks*.
(*Camera-ready version of the paper will be uploaded soon*)

Special thanks and acknowledgement to [@greentfrapp](https://github.com/greentfrapp/attention-primer) for his implementation of
the self-attention mechanism!

## Model Description

The Attention for Network Intrusion Detection (ANID) model draws inspiration from the transformer model by [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
The ANID is a simple feed-forward network that does sequence-to-sequence predictions without the encoder-decoder structure used in the transformer.

Block diagram of the ANID model:

<div>
<img src="https://raw.githubusercontent.com/Tanmengxuan/ANID/master/images/anid.png" alt="anid" width="whatever" height="550px" align="center" style="display: block;">
</div>

