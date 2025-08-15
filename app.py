# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from janome.tokenizer import Tokenizer

st.title("Self-Attention 可視化アプリ")
st.markdown("文章を入力して「解析する」を押すと、Self-Attention の重みを可視化します。")

# 文章入力
text = st.text_area("文章を入力してください", "嵐は去ったが、街の傷跡は深く残っていた。")

if st.button("解析する"):
    if text.strip() == "":
        st.warning("文章を入力してください")
    else:
        # Janome で形態素解析
        t = Tokenizer()
        tokens = [token.surface for token in t.tokenize(text)]
        st.write("トークン分割結果:", tokens)

        # Self-Attention 計算
        seq_len = len(tokens)
        torch.manual_seed(0)
        x = torch.randn(seq_len, 16)  # 埋め込み次元 16

        Q = x
        K = x
        V = x

        d_k = Q.size(-1) ** 0.5
        scores = torch.matmul(Q, K.T) / d_k
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 可視化
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(attn_weights.detach().numpy(),
                    annot=True,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="YlGnBu",
                    ax=ax)
        plt.xlabel("Key（見てる単語）")
        plt.ylabel("Query（注目してる単語）")
        plt.title("Self-Attention Weights")
        st.pyplot(fig)

plt.figure(figsize=(len(tokens), len(tokens)))  # トークン数に応じて動的にサイズ変更
        sns.heatmap(attn_weights.detach().numpy(), annot=True, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu")
        st.pyplot(plt)



