This is to provide a flexible and simple code encapsulation for Wenet.
Based on Wenet-3.1.

+ The following deficiencies exist in the official code:
  1. Only fbank feature calculation.
  2. The data reading method is incorrect.
  3. There are fewer callable functions and they are not very flexible.
  4. You need to convert the model to JIT mode before you can perform inference.

pip install git+https://github.com/ToughmanL/wenet-encapsulation.git


