| benchmark | rows | English in SFT | pair in SFT | `clean` rows | languages with the pair in SFT | English in an RL prompt (any language / own language) | test split used in reward fitting or selection |
|:------------|------:|--------:|--------:|--------:|:--------------|------------:|:--------------------|
| FLORES+ devtest | 45,540 | 95.3% | 4.1% | 4.7% | hau 83%, ibo 55%, yor 35% (7 languages) | 0.69% / 0.05% | devtest gaps chose the 5 NLLB languages (final run) |
| NTREX-128 | 47,928 | 39.9% | 1.2% | 60.1% | hau 21%, yor 7% | 0.45% / 0.08% | final run: anchors |
| MAFAND-MT test | 20,009 | 21.9% | 15.5% | 78.1% | ibo 90%, yor 48% (8 languages) | 0.46% / 0.16% | no (MAFAND dev: Pidgin, monitoring) |
| SmolSent | 38,832 | 91.2% | 5.5% | 8.8% | wol 73%, bam 65%, ewe 61% | 0.23% / 0.08% | final run: anchors |
| GlotStoryBook | 3,426 | 7.2% | 0.0% | 92.8% | none | 1.7% / 0.09% | final run: anchors |
| SALT test | 2,500 | 84.4% | 16.8% | 15.6% | lug 84% | 0.2% / 0.0% | final run: anchors |
| IbomMT test | 2,024 | 95.3% | 0.0% | 4.7% | none | 0.69% / 0.10% | RL-final and final run: Efik thresholds (997 of 1,012); final run: anchors |
| FLORES-MOZ | 1,012 | 95.3% | 0.0% | 4.7% | none | 0.69% / 0.0% | final run: anchors |
