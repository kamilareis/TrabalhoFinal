from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_test', methods=['GET', 'POST'])
def train_test():
    if request.method == 'POST':

        classifier_name = request.form['classifier']
        n_neighbors = int(request.form['n_neighbors'])

        classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=n_neighbors),
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        model = classifiers[classifier_name]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        class_labels = [0, 1, 2]

        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        plt.xticks(class_labels, class_labels, rotation=45)
        plt.yticks(class_labels, class_labels)
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('results.html', accuracy=accuracy, f1_macro=f1_macro, plot_url=plot_url)

    return render_template('train_test.html')

if __name__ == '__main__':
    app.run(debug=True)