from django.http import HttpResponse
from django.template import loader
import requests
from scrapy.http import TextResponse
import pandas as pd
import math, nltk, warnings
from sklearn.neighbors import NearestNeighbors
import sys
import tweepy
from textblob import TextBlob
from fuzzywuzzy import fuzz



reload(sys)
sys.setdefaultencoding('utf8')
warnings.filterwarnings('ignore')
PS = nltk.stem.PorterStemmer()
import matplotlib
matplotlib.use('GTKAgg')
MovieUrlAndDescrption = pd.read_csv("F:\Movie Recommendation\data\MovieUrlAndDescrption.csv")
MoviesNameUrl = MovieUrlAndDescrption.set_index('name').T.to_dict('list')
gaussian_filter = lambda x, y, sigma: math.exp(-(x - y) ** 2 / (2 * sigma ** 2))
def entry_variables(df, id_entry):
    col_labels = []
    if pd.notnull(df['director_name'].iloc[id_entry]):
        for s in df['director_name'].iloc[id_entry].split('|'):
            col_labels.append(s)

    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i + 1))
        if pd.notnull(df[column].iloc[id_entry]):
            for s in df[column].iloc[id_entry].split('|'):
                col_labels.append(s)

    if pd.notnull(df['plot_keywords'].iloc[id_entry]):
        for s in df['plot_keywords'].iloc[id_entry].split('|'):
            col_labels.append(s)
    return col_labels
def add_variables(df, REF_VAR):
    for s in REF_VAR: df[s] = pd.Series([0 for _ in range(len(df))])
    colonnes = ['genres', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'plot_keywords']
    for categorie in colonnes:
        for index, row in df.iterrows():
            if pd.isnull(row[categorie]): continue
            for s in row[categorie].split('|'):
                if s in REF_VAR: df.set_value(index, s, 1)
    return df
def recommand(df, id_entry):
    df_copy = df.copy(deep=True)
    liste_genres = set()
    for s in df['genres'].str.split('|').values:
        liste_genres = liste_genres.union(set(s))
    variables = entry_variables(df_copy, id_entry)
    variables += list(liste_genres)
    df_new = add_variables(df_copy, variables)
    X = df_new.as_matrix(variables)
    nbrs = NearestNeighbors(n_neighbors=31, algorithm='auto', metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    xtest = df_new.iloc[id_entry].as_matrix(variables)
    xtest = xtest.reshape(1, -1)
    distances, indices = nbrs.kneighbors(xtest)
    return indices[0][:]
def extract_parameters(df, liste_films):
    parametres_films = ['_' for _ in range(31)]
    i = 0
    max_users = -1
    for index in liste_films:
        parametres_films[i] = list(df.iloc[index][['movie_title', 'title_year',
                                                   'imdb_score', 'num_user_for_reviews',
                                                   'num_voted_users']])
        parametres_films[i].append(index)
        max_users = max(max_users, parametres_films[i][4])
        i += 1

    title_main = parametres_films[0][0]
    annee_ref = parametres_films[0][1]
    parametres_films.sort(key=lambda x: critere_selection(title_main, max_users,
                                                          annee_ref, x[0], x[1], x[2], x[4]), reverse=True)
    return parametres_films
def sequel(titre_1, titre_2):
    if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
        return True
    else:
        return False
def critere_selection(title_main, max_users, annee_ref, titre, annee, imdb_score, votes):
    if pd.notnull(annee_ref):
        facteur_1 = gaussian_filter(annee_ref, annee, 20)
    else:
        facteur_1 = 1
    sigma = max_users * 1.0
    if pd.notnull(votes):
        facteur_2 = gaussian_filter(votes, max_users, sigma)
    else:
        facteur_2 = 0
    if sequel(title_main, titre):
        note = 0
    else:
        note = imdb_score ** 2 * facteur_1 * facteur_2
    return note
def add_to_selection(film_selection, parametres_films):
    film_list = film_selection[:]
    icount = len(film_list)
    for i in range(31):
        already_in_list = False
        for s in film_selection:
            if s[0] == parametres_films[i][0]: already_in_list = True
            if sequel(parametres_films[i][0], s[0]): already_in_list = True
        if already_in_list: continue
        icount += 1
        if icount <= 6:
            film_list.append(parametres_films[i])
    return film_list
def remove_sequels(film_selection):
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue
            if sequel(film_1[0], film_2[0]):
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)
    film_list = [film for film in film_selection if film[0] not in removed_from_selection]
    return film_list
def find_similarities(df, id_entry, del_sequels=True, verbose=False):
    liste_films = recommand(df, id_entry)
    parametres_films = extract_parameters(df, liste_films)
    film_selection = []
    film_selection = add_to_selection(film_selection, parametres_films)
    if del_sequels: film_selection = remove_sequels(film_selection)
    film_selection = add_to_selection(film_selection, parametres_films)
    selection_titres = []
    for i, s in enumerate(film_selection):
        selection_titres.append([s[0],s[2]])
    return selection_titres
def recommend1(df, Name):
    aDict = {}
    for i in range(len(df['movie_title'])):
        aDict[df['movie_title'][i]] = i
    b = find_similarities(df, aDict[Name], del_sequels=True, verbose=True)
    name=[b[0][0],b[1][0],b[2][0],b[3][0],b[4][0],b[5][0]]
    score=[b[0][1], b[1][1], b[2][1], b[3][1], b[4][1],b[5][1]]
    return name,score
def getimg(movies):
    l1=list()
    for a in movies:
        c =MoviesNameUrl[a][0]
        l1.append(c)
    return l1

def sentiment(movie_name):
    auth = tweepy.OAuthHandler("1StJcHu1vhWyOH1T4pazEpqp1","2T14LRZ249RYp4YvR3plnp8IH8D4sSSaoAXOgq6NPI0ZUwzdMd")
    auth.set_access_token("478707834-XZz0hLXQhLgkJDPSMJQ3tBrdxoABdga82Wg1G3hO","wbsw5pq4qIb0LBytO5vglaqgQVXystzEFDucNKimUTUzU")
    api = tweepy.API(auth)
    N = 100
    Tweets = tweepy.Cursor(api.search, q=movie_name).items(N)
    neg = 0.0
    pos = 0.0
    neg_count = 0
    neutral_count = 0
    pos_count = 0
    for tweet in Tweets:
        blob = TextBlob(tweet.text)
        if blob.sentiment.polarity < 0:
            neg += blob.sentiment.polarity
            neg_count += 1
        elif blob.sentiment.polarity == 0:
            neutral_count += 1
        else:
            pos += blob.sentiment.polarity
            pos_count += 1
    return (pos_count,neutral_count,neg_count)

def getimgdec(movies):
    try:
        r = requests.get('http://www.imdb.com/find?ref_=nv_sr_fn&q=' + movies)
        response = TextResponse(r.url, body=r.text, encoding='utf-8')
        spcial = response.css('.findList .findResult .result_text a::attr(href)').extract_first()
        spcial = 'http://www.imdb.com/' +spcial.encode("ascii", "ignore")
    except AttributeError:
        spcial = 'http://www.imdb.com/'
    return MoviesNameUrl[movies][1],MoviesNameUrl[movies][0],spcial

def index(request):
    template = loader.get_template("data/index.html")
    df = pd.read_csv("movie_metadata.csv", encoding='latin-1')
    df.info(verbose=False)
    df['movie_title'] = df['movie_title'].apply(
        lambda x: x.decode("utf-8").replace(u'\xa0', u' ').replace(u'\xc2', u' ').encode("ascii", "ignore").strip())
    df.reset_index(inplace=True, drop=True)
    list1=MoviesNameUrl.keys()
    if request.method == 'POST':
        entry = request.POST['name']
        name,score = recommend1(df,entry)
        l1 = getimg(name)
        pos_count, neutral_count, neg_count=sentiment(entry)
        img_des, image_url, movieurl = getimgdec(entry)
        contex = {
            "type": "2",
            "result": zip(name,score, l1),
            "pos_count":pos_count,
            "neutral_count":neutral_count,
            "neg_count":neg_count,
            "img_des": img_des,
            "image_url": image_url,
            "movieurl" : movieurl,
            "dataset": entry,
            "list":list1
        }
    else:
        contex = {
            "type": "3",
            "result": "nan",
            "pos_count":"nan",
            "neutral_count":"nan",
            "neg_count":"nan",
            "img_des": "nan",
            "image_url": "nan",
            "dataset": "nan",
            "list": list1
        }
    return HttpResponse(template.render(contex, request))

