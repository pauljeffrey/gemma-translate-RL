| language | RL on top of SFT: $\Delta$ chrF++, `no\_pair` rows | extra SFT data: C3 against A2W | parallel tokens added to the SFT stage | other |
|:--|:--|:--|:--|:--|
| Amharic | +4.4 (FLORES+), +4.2 (NTREX), +4.1 (MAFAND), +3.4 (SmolSent) | +0.02 (weak dev, 60 sent.) | 1.5 M + 0.37 M SA + 0.14 M code-switch (B_v2); 4.47 M NLLB teacher tokens over 27 languages (C3) | paper-level quality fell with B_v2_ibom (20.5 to 18.8 / 17.6) |
| Tigrinya | +4.3 (FLORES+), +4.5 (NTREX), +3.6 (SmolSent) | +0.55 (weak dev, 60 sent.) | 2.0 M (B_v2), plus C3 teacher data | B_v2_ibom against RL model: +0.70 / +0.74 at $\alpha$ 0.5 / 0.7 |
