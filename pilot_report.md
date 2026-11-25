# Pilot report

- model: `google/gemma-2-2b-it` (26 layers, torch.float16)
- items: 100 over 40 relations
- context-following rate: 0.280
- detection position (primary): end_of_context

## Headline

| method                       | detector        |   auroc |   auroc_base_rate |   auroc_over_readout | det_position   | steerer         |   best_factor |   flip_rate |   flip_rate_flippable |   flip_rate_raw |   d_margin |   specific_effect |   fluency |
|:-----------------------------|:----------------|--------:|------------------:|---------------------:|:---------------|:----------------|--------------:|------------:|----------------------:|----------------:|-----------:|------------------:|----------:|
| prompting                    | p_true          |  0.7999 |               0.5 |             nan      |                | prompt_instruct |           0   |        0.03 |                0.75   |            0.24 |    -4.559  |           -4.559  |   -0.3707 |
| CAD                          | context_kl      |  0.5881 |               0.5 |             nan      |                | cad             |           2   |        0.95 |                0.9896 |            0.56 |    41.1914 |           41.1914 |  nan      |
| AdaCAD                       | context_jsd     |  0.569  |               0.5 |             nan      |                | adacad          |           2   |        0.41 |                0.4271 |            0.33 |    16.5542 |           16.5542 |  nan      |
| CK-PLUG                      | confidence_gain |  0.3577 |               0.5 |             nan      |                | ckplug          |           0   |        0.01 |                0.25   |            0.02 |     4.8538 |            4.8538 |  nan      |
| CAA                          | diffmean_proj   |  0.6105 |               0.5 |               0.0532 | end_of_context | caa             |           0.5 |        0.04 |                1      |            0.22 |    -9.1645 |           -5.8742 |   -4.0623 |
| CAA (orthogonalized)         | diffmean_proj   |  0.6105 |               0.5 |               0.0532 | end_of_context | act_add         |           0.5 |        0.04 |                1      |            0.2  |    -8.2865 |           -4.9955 |   -4.1338 |
| BOW baseline                 | bow             |  0.5496 |               0.5 |             nan      |                |                 |         nan   |      nan    |              nan      |          nan    |   nan      |          nan      |  nan      |
| margin (DV floor)            | margin          |  0.5358 |               0.5 |             nan      |                |                 |         nan   |      nan    |              nan      |          nan    |   nan      |          nan      |  nan      |
| logit lens (readout control) | logit_lens      |  0.5572 |               0.5 |             nan      | end_of_context |                 |         nan   |      nan    |              nan      |          nan    |   nan      |          nan      |  nan      |

## Reading it

- `auroc` beats `auroc_base_rate` or the detector is reading the
  relation, not the item.
- `auroc` beats the **BOW baseline** row or it is reading the text,
  not the model.
- `auroc_over_readout` > 0 or the probe is decoding an answer the
  model had already committed to.
- `specific_effect` is delta-margin minus the matched control;
  `flip_rate_flippable` conditions on items that could move.
- `flip_rate` vs `flip_rate_raw`: a large gap means the flips are
  an artifact of the passage template, not the intervention.
