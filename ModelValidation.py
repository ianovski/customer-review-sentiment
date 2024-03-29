from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class ModelValidation:

  # author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
  #             Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
  #             Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
  #             and Jaques Grobler and Robert Layton and Jake VanderPlas and
  #             Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
  # title     = {{API} design for machine learning software: experiences from the scikit-learn project},
  # booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
  # year      = {2013},
  # pages = {108--122}
  # Availability: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
  def plot_confusion_matrix(self,y_true, y_pred, classes,
                           normalize=False,
                           title=None,
                           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

  def plot_result_comparison(self, y_true, y_pred, classes, title):
    barWidth = 0.25
    y_true_count = y_true.value_counts()
    y_pred_count = y_pred.value_counts()
    r1 = np.arange(len(classes))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1,y_true_count,width=barWidth,edgecolor='white',label='True')
    plt.bar(r2,y_pred_count,width=barWidth,edgecolor='white',label='Predicted')
    y_true_count = y_true.value_counts()
    
    plt.xlabel('Sentiment Labels')
    plt.ylabel('Label Count')
    plt.title(title)
    plt.xticks([r + barWidth for r in range(len(classes))], classes)
    # Create legend & Show graphic
    plt.legend()
    plt.savefig('figures/result_comparison.png')
    plt.show()

