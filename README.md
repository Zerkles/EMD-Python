# Przewidywanie oceny na podstawie recenzji

## EMD - Projekt w Python'ie

Autor: Wojciech Lulek, index: 136280

## Wstęp

Podstawowym celem projektu jest predykcja oceny w skali 1-5 na podstawie pozostałych cech recenzji, jakich jak treść,
podusmowanie czy głosy ([nie]pomocny) od użytkowników serwisu. Problem jest zdecydowanie nietrywialny a
satysfakcjonujące wyniki niełatwe do osiągnięcia.

## Instrukcja uruchamiana:

Projekt składa się z 3 notebook'ów oraz zbioru danych. Powinny być one uruchamiane w kolejności:

1. preprocessing.ipynb
2. process_BoW.ipynb/process_w2v.ipynb
3. machine_learning.ipynb

Po uruchomieniu notebook'ów z podpunktów 1 oraz 2 zostaną wygenerowane nowe pliki .csv zawierające wygenerowane cechy.

Dodatkowo został utworzony notebook "testing.ipynb" służący do uruchamiana na dodatkowym zbiorze. Aby tego dokonać
należy zmodyfikować ścieżkę do pliku testowego w pierwszej komórce z kodem, następnie automatycznie zostanie wczytany
wytrenowany klasyfikator oraz dokonana ocena algorytmu.

## Metodologia

W celu wykonania przetestowano dwa podejścia do reprezentacji tekstu dla zadania klasyfikacji, są to:

* Bag Of Words - wykonane za pomocą CountVectorizer z biblioteki sklrearn
* Word2Vector - wykonana za pomocą biblioteki gensim oraz gotowego modelu "GoogleNews-vectors-negative300"

Parametry obu modeli zostały dostosowane do problemu w odpowiadających im notebook'ach.

Do zadania klasyfikacji zdecydowano się wybrać RandomTreeClassifier z biblioteki sklrean głównie z powodu dobrej
wydajności tego algorytmu. Optymalne hiperparametry zostały ustalone za pomocą GridSearchCV (ponownie sklearn), który
umożliwia przetestowanie oraz porównanie wielu zestawów hiperparametrów.

## Rezultaty:

## Wnioski

Z pewnością dużym ograniczeniem przy wykonywaniu projektu były posiadane zasoby sprzętowe. W przypadku posiadania
bardziej wydajnej maszyny możnaby przetestować większą ilość hiperparametrów oraz klasyfikatorów.

Odnosząc się do zastosowanego podejścia do przetwarzania cech, wykorzystane rozwiązanie nie jest idealnie ponieważ cechy
natywne wymieszane są z reprezentacjami tekstu (BoW lub W2V) co powoduje problemy z normalnizacją a także cechy uzyskane
z przetwarzania tekstu są o wiele liczniejsze niż pozostałe cechy co sprawia że natywne cechy mają dużo mniejszy wpływ
na uzyskany model niż powinny. W celu poprawy tego zjawiska możnaby wykorzystać np. zespół klasyfikatorów.

Wartościowym byłoby również wykonanie undersamplingu w celu zrównoważenie ilości przykładów w klasach.

Ciekawym byłoby również przetestowania sieci nauronowych do predykcji zamiast klasycznych klasyfikatorów.

