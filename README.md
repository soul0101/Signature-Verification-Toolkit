# Signature-Verification-Toolkit

The Signature Verification Toolkit Daisi applies modern deep learning techniques in addressing the task of offline signature verification -
given a pair (or pairs of) signatures, determine if they are produced by the same user (genuine signatures) or different users (potential forgeries). 

![image](https://user-images.githubusercontent.com/53980340/196192992-5bace81e-7485-4a9e-87aa-50b92c1de122.png)

## Signature Verification Toolkit Functions

### Detector

Returns a list of bounding boxes where signatures are located in an image.

```python
import pydaisi as pyd
signature_verification_toolkit = pyd.Daisi("soul0101/Signature Verification Toolkit")
boxes, scores, classes, detections = signature_verification_toolkit.signature_detector(img_tensor).value
```

![image](https://user-images.githubusercontent.com/53980340/196195210-3d66dd8d-c010-4cfc-8dbe-091ca46b3b98.png)

### Cleaner
This function takes in a list of signatures, or a single signature and returns the cleaned signature images (removal of background lines and text).

```python
import pydaisi as pyd
import numpy as np
from PIL import Image
signature_verification_toolkit = pyd.Daisi("soul0101/Signature Verification Toolkit")

sign_np = np.array(Image.open(<path/to/image>).convert(mode='RGB'))
cleaned_sign = signature_verification_toolkit.signature_cleaner(orig_sign_np).value
```

![image](https://user-images.githubusercontent.com/53980340/196195344-bb1268d0-dac5-4d09-be1a-395a184bc46b.png)


### Matcher

Returns a distance measure given a pair of signatures

```python
import pydaisi as pyd
signature_verification_toolkit = pyd.Daisi("soul0101/Signature Verification Toolkit")

result = signature_verification_toolkit.verify_signatures(orig_sign_np, check_sign_np).value
```

![image](https://user-images.githubusercontent.com/53980340/196196160-681c073b-e95b-4e3b-9562-99064c7ed73d.png)

### Credits
- https://github.com/victordibia/signver
- <a href="https://www.freepik.com/free-vector/signing-contract-concept-illustration_12325314.htm#query=signature&position=45&from_view=search&track=sph">Image by storyset</a> on Freepik
