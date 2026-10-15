# Training inputs and display geometry

The actual static/video benchmark first converts each decoded full-resolution frame to grayscale, then directly resizes 1,936 × 1,216 to 64 × 64 using area interpolation. This is anisotropic resizing: the relative horizontal-to-vertical scale is 1,216/1,936 = 0.6281. It explains the apparent vertical elongation in the original model-input illustrations.

For both fixed examples, the original RGB midpoint frame independently reproduces the cached Single-frame tensor with maximum absolute error 0. The independently re-created ECC-registered Mean-image cache also has maximum absolute error 0. The complete cache audit verifies the recorded midpoint, uniform sequence-sampling and signed-difference identities.

At the author's request, the revised Fig. 5b instead displays source-aspect illustrations. The cached square-grid contents are reprojected to a 64 × 40 content region, then centred in a 64 × 64 display canvas with 12 pixels of background padding above and below. Pixel rounding approximates the source aspect ratio. These illustrated arrays are not the inputs on which the reported models were trained or evaluated.

The native reference frames retain their original pixels and aspect ratio. The prediction table and every numerical panel retain the original benchmark predictions. No new letterbox inference or training has taken place. A future aspect-preserving model comparison must rebuild inputs and retrain every affected comparator on the same fixed splits; simply changing a figure cannot establish that result.
