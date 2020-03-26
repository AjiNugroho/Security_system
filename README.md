# NSFW Detection Machine Learning Model
Trained on 60+ Gigs of data to identify:
- `drawings` - safe for work drawings (including anime)
- `hentai` - hentai and pornographic drawings
- `neutral` - safe for work neutral images
- `porn` - pornographic images, sexual acts
- `sexy` - sexually explicit images, not pornography


## Current Status:
93% Accuracy with the following confusion matrix, based on Inception V3.

## Requirements:
keras (tested with versions > 2.0.0)

tensorflow (Not specified in setup.py)

## Usage
```python
from nsfw_detector import NSFWDetector
detector = NSFWDetector('./nsfw.299x299.h5')
detector_mobilenet = NSFWDetector('./nsfw_mobilenet2.224x224.h5')

# Predict single image
detector.predict('2.jpg')
# {'2.jpg': {'sexy': 4.3454722e-05, 'neutral': 0.00026579265, 'porn': 0.0007733492, 'hentai': 0.14751932, 'drawings': 0.85139805}}

# Predict multiple images at once using Keras batch prediction
detector.predict(['/Users/bedapudi/Desktop/2.jpg', '/Users/bedapudi/Desktop/6.jpg'], batch_size=32)
# {'2.jpg': {'sexy': 4.3454795e-05, 'neutral': 0.00026579312, 'porn': 0.0007733498, 'hentai': 0.14751942, 'drawings': 0.8513979}, '6.jpg': {'drawings': 0.004214506, 'hentai': 0.013342537, 'neutral': 0.01834045, 'porn': 0.4431829, 'sexy': 0.5209196}}

# Predict single image using mobilenet
detector_mobilenet.predict('test.jpg', image_size=(224,224))
# {'test.jpg': {'drawings': 9.211938e-05, 'hentai': 0.0047431793, 'sexy': 0.052986998, 'neutral': 0.05684011, 'porn': 0.8853376}}
