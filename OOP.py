from imports import * 

class Sentiment_Analysis:

    def __init__(self,datafile):
        self.datafile = datafile

    def read_data(self,datafile):
        dataset = pd.read_csv(datafile)
        dataset = dataset[["text", "airline_sentiment"]]
        return dataset

    def visualize_data(self,sentiment, dataset):
        if (sentiment == "negative"):
          print(f'Negative Sentiment Tweets: {dataset["airline_sentiment"].value_counts()["negative"]}')
          negative = dataset[dataset['airline_sentiment']=='negative']
          words = ' '.join(negative['text'])

          wordcloud = WordCloud(background_color='white',width=3000,height=2500).generate(words)
          plt.figure(1,figsize=(12, 12))
          plt.imshow(wordcloud)
          plt.axis('off')
          plt.show()
        elif(sentiment == "positive"):
          print(f'Positive Sentiment Tweets: {dataset["airline_sentiment"].value_counts()["positive"]}')
          positive = dataset[dataset['airline_sentiment']=='positive']
          words = ' '.join(positive['text'])
          wordcloud = WordCloud(background_color='white',width=3000,height=2500).generate(words)
          plt.figure(1,figsize=(12, 12))
          plt.imshow(wordcloud)
          plt.axis('off')
          plt.show()


    def tweet_to_words(self,tweet):

        letters_only = re.sub("[^a-zA-Z]", " ",tweet) 
        words = letters_only.lower().split()                             
        stops = set(stopwords.words("english"))                  
        meaningful_words = [w for w in words if not w in stops] 

        return( " ".join( meaningful_words )) 

    def clean_data(self,dataset):
      return dataset['text'].apply(lambda x: self.tweet_to_words(x))

    def tranform_to_vector(self,dataset):
        X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['airline_sentiment'], test_size=0.2, random_state=42)
        v = CountVectorizer(analyzer = "word")
        train_features= v.fit_transform(X_train)
        test_features=v.transform(X_test)
        return train_features,test_features, y_train, y_test

    def train_model(self, model, train_features,test_features, y_train, y_test):
        if(model=="random forest"):
          rfc_model = RandomForestClassifier(n_estimators=200)
          rfc_model.fit(train_features,y_train)
          return rfc_model
        elif(model=="decision tree"):
          dtc_model = DecisionTreeClassifier()
          dtc_model.fit(train_features,y_train)
          return dtc_model

    def predict(self, model, test_features):
        pred = model.predict(test_features)
        return pred

    def evaluate_model(self,pred, test_y):
        accuracy = accuracy_score(pred,test_y)
        print('Accuracy:'+str(accuracy))
        print(classification_report(pred,test_y))

    def save_to_pkl(self,model):
        pickle.dump(model, open(f'{model}.pkl', 'wb'))

sa = Sentiment_Analysis("airline_sentiment_analysis.csv")
dataset = sa.read_data(sa.datafile)
dataset['text'] = sa.clean_data(dataset)

sa.visualize_data("negative", dataset)

sa.visualize_data("positive", dataset)

train_features,test_features, y_train, y_test = sa.tranform_to_vector(dataset)
model = sa.train_model("random forest",train_features,test_features, y_train, y_test)
preds = sa.predict(model,test_features)

sa.evaluate_model(preds,y_test)

sa.save_to_pkl(model)

