# Pied Piper NotAHotDog

Image classification using TensorFlow (Inception v3) that labels any JPEG image as "hot dog" or "not a hot dog".

Inspired by Jian Yang's "Shazaam for food" from Silicon Valley: Season 4 Episode 4.

![Not a hot dog](https://media.giphy.com/media/3ohzdXIKl0BjNK2g3m/giphy.gif)

## Instructions

1. Clone repository.

``git clone https://github.com/TheDoctor0/NotAHotDog.git``
    
2. Install requirements.

``python -m pip install -r requirements.txt``

3. Get Tensorflow Inception v3.

``curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py``

4. Retrain model.

``python retrain.py --bottleneck_dir=bottlenecks --model_dir=inception --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=images``

5. Use classifier on any JPEG image.

``python checker.py image_path.jpg``

### Example
```
$ python checker.py dog.jpg
not a hot dog (98.92%)
```
