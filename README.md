# Przewidywanie oceny na podstawie recenzji

## EMD - Projekt w Python'ie

Autor: Wojciech Lulek, index: 136280

## Wstęp

Podstawowym celem projektu jest predykcja oceny w skali 1-5 na podstawie pozostałych cech recenzji, jakich jak treść,
podusmowanie czy głosy ([nie]pomocny) od użytkowników serwisu. Problem jest zdecydowanie nietrywialny a
satysfakcjonujące wyniki niełatwe do osiągnięcia.

## Instrukcja uruchamiana:
Aby uruchomić i przetestować model należy:
1. Pobrać niezbędne pliki z (), znajdują się tam wektory słów dla word2vec oraz inne niezbędne pliki
2. Uruchomić notebook preprocessing.ipynb
3. Uruchomić plik przeznaczony do testowania, testing.py
   1. W sekcji main należy ustawić dwie wartości:
   2. representation - do wyboru z [bow,w2v]
   3. test_data_path - ścieżka do pliku z przykładami testowymi


Projekt dodatkowo zawiera 3 notebook'i oraz zbiór danych służące w trakcie realizacji zadania. Notebooki powinny być uruchamiane w kolejności:

1. preprocessing.ipynb - przetwarzanie oraz analiza danych
2. process_bow.ipynb/process_w2v.ipynb - przetwarzanie dokumentów na odpowiednie reprezentacje
3. machine_learning.ipynb - uczenie maszynowe oraz poszukiwanie hiperparametrów

Po uruchomieniu notebook'ów z podpunktów 1 oraz 2 zostaną wygenerowane nowe pliki .csv zawierające wygenerowane cechy.

## Metodologia

W celu wykonania przetestowano dwa podejścia do reprezentacji tekstu dla zadania klasyfikacji, są to:

* Bag Of Words - wykonane za pomocą CountVectorizer z biblioteki sklrearn
* Word2Vector - wykonana za pomocą biblioteki gensim oraz gotowego modelu "GoogleNews-vectors-negative300"

Parametry obu modeli zostały dostosowane do problemu w odpowiadających im notebook'ach.

Do zadania klasyfikacji zdecydowano się wybrać RandomTreeClassifier z biblioteki sklrean głównie z powodu dobrej
wydajności tego algorytmu. Optymalne hiperparametry zostały ustalone za pomocą GridSearchCV (ponownie sklearn), który
umożliwia przetestowanie oraz porównanie wielu zestawów hiperparametrów.

## Rezultaty:

Wyniki dla klasyfikatora wytrenowanego przy pomocy reprezentacji word2vec:
=================== Results: RandomClassifier ===================
                1.0       2.0       5.0       4.0       3.0
F1         0.140262  0.093050  0.286117  0.204460  0.147258
Precision  0.107968  0.060610  0.506922  0.209687  0.115887
Recall     0.200116  0.200207  0.199304  0.199487  0.201919
=================== Results: MajorityClassifier =================
           1.0  2.0       5.0  4.0  3.0
F1         0.0  0.0  0.672355  0.0  0.0
Precision  0.0  0.0  0.506427  0.0  0.0
Recall     0.0  0.0  1.000000  0.0  0.0
=================== Results: RandomForestClassifier =============
                1.0       2.0       5.0       4.0       3.0
F1         0.743568  0.602527  0.721242  0.544872  0.620299
Precision  0.676338  0.442269  0.872078  0.496645  0.592386
Recall     0.825640  0.944924  0.614890  0.603472  0.650974

Wyniki dla klasyfikatora wytrenowanego przy pomocy reprezentacji BagOfWords:
=================== Results: RandomClassifier ===================
                1.0       2.0       5.0       4.0       3.0
F1         0.140262  0.093050  0.286117  0.204460  0.147258
Precision  0.107968  0.060610  0.506922  0.209687  0.115887
Recall     0.200116  0.200207  0.199304  0.199487  0.201919
=================== Results: MajorityClassifier =================
           1.0  2.0       5.0  4.0  3.0
F1         0.0  0.0  0.672355  0.0  0.0
Precision  0.0  0.0  0.506427  0.0  0.0
Recall     0.0  0.0  1.000000  0.0  0.0
=================== Results: RandomForestClassifier =============
                1.0       2.0       5.0       4.0       3.0
F1         0.419865  0.196455  0.521490  0.253587  0.272205
Precision  0.349536  0.164208  0.733387  0.307853  0.184898
Recall     0.525624  0.244464  0.404592  0.215585  0.515730

Podsumowując, lepsze wyniki uzyskano wykorzystując reprezentację word2vec.
## Wnioski

Z pewnością dużym ograniczeniem przy wykonywaniu projektu były posiadane zasoby sprzętowe. W przypadku posiadania
bardziej wydajnej maszyny możnaby przetestować większą ilość hiperparametrów oraz klasyfikatorów.

Odnosząc się do zastosowanego podejścia do przetwarzania cech, wykorzystane rozwiązanie nie jest idealnie ponieważ cechy
natywne wymieszane są z reprezentacjami tekstu (BoW lub W2V) co powoduje problemy z normalnizacją a także cechy uzyskane
z przetwarzania tekstu są o wiele liczniejsze niż pozostałe cechy co sprawia że natywne cechy mają dużo mniejszy wpływ
na uzyskany model niż powinny. W celu poprawy tego zjawiska możnaby wykorzystać np. zespół klasyfikatorów.

Ciekawym byłoby również przetestowania sieci nauronowych do predykcji zamiast klasycznych klasyfikatorów.

