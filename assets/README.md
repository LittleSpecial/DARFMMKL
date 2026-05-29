# DARFMMKL Assets

Supplementary assets for the DARFMMKL paper (ICML 2026). Not included in the
camera-ready PDF — kept here for future use (talks, posters, slide decks).

## Files

### `darfmmkl_overview_clean.png`

- **Resolution**: 2816 × 1536 px (RGBA, ~6 MB)
- **Source**: Gemini image generation, May 2026
- **Status**: Not included in CR paper (DARFMMKL paper follows MKL-literature
  convention of no Figure 1 pipeline diagram). Algorithm 1 and the math
  formulation cover the same content in the body.
- **When to use**:
  - Conference talk / oral presentation cover slide
  - Poster session main figure
  - Blog post / social media announcement of the paper
  - Future extended journal version where a Figure 1 might be added

## Figure description

Four-stage horizontal pipeline:
1. **Candidate Kernel Pool** — Traditional kernels (RBF, Polynomial) plus
   RFM kernels with feature-importance matrix W learned via AGOP
   fixed-point iteration.
2. **Nyström Sketching for Scalability** — Candidate kernel K (n × n) is
   approximated by Nyström sketch K̃ (s × s, s ≪ n), decoupling cost
   from sample size n.
3. **Joint Diversity-Quality Selection** — Compute diversity matrix G
   (pairwise prediction disagreement) and quality vector b (CKA with
   ideal kernel ỹỹ^T); maximize η^T G η + λ b^T η subject to
   η ∈ {0,1}^M, Σ η_i = m. NP-hard binary quadratic program.
4. **Scalable Kernel Selection via Glover Linearization** — BQP → LP
   via Glover linearization with continuous relaxation; take top-m to
   define the selected kernel set; final composite kernel
   κ*(x, x') = (1/m) Σ κ_i(x, x') is fed to an SVM classifier.

## Prompt used to generate this figure

See the project notes or `/Users/zhaoxu/.claude/skills/paper-diagram-prompt/`
for the prompt template. The specific filled-in prompt for DARFMMKL is
preserved in the conversation log from May 2026.

The prompt was based on
[Leey21/awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
(论文架构图 section), adapted for DARFMMKL's specific methodology.
