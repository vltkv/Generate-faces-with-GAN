# Dokumentacja: Generowanie twarzy przy użyciu GAN

## Spis treści
1. [Konfiguracja środowiska](#konfiguracja-środowiska)
2. [Konfiguracja Kaggle](#konfiguracja-kaggle)
3. [Uruchamianie modelu](#uruchamianie-modelu)
4. [Flagi wiersza poleceń](#flagi-wiersza-poleceń)
5. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Konfiguracja środowiska

```bash
# klonowanie repozytorium
git clone https://github.com/user/Generate-faces-with-GAN.git
cd Generate-faces-with-GAN

# utworzenie wirtualnego środowiska
python -m venv venv

# aktywacja w Windows
venv\Scripts\activate
# aktywacja w Linux/Mac:
source venv/bin/activate

# instalacja zależności
pip install --upgrade pip

pip install -r requirements.txt
```

## Konfiguracja Kaggle

Należy utworzyć konto na [Kaggle](https://www.kaggle.com/), wygenerować API Token, umieścić pobrany `kaggle.json` w `~/.kaggle`, a następnie zmienić uprawnienia na `chmod 600`.


## Uruchamianie modelu

### Podstawowe uruchomienie

```bash
python main.py
```

Domyślnie, program będzie szukał danych w katalogu `./data/celeba`. Jeśli dane nie zostały wcześniej pobrane, można użyć flagi `--use_kagglehub` do pobrania danych z Kaggle.

## Flagi CLI

Można dostosować proces uczenia przy użyciu następujących flag:

| Flaga | Typ | Domyślna wartość | Opis |
|-------|-----|------------------|------|
| `--dataroot` | string | './data/celeba' | Ścieżka do katalogu z danymi |
| `--batch_size` | int | 128 | Rozmiar partii danych podczas treningu |
| `--image_size` | int | 64 | Rozmiar przestrzenny obrazów treningowych |
| `--num_epochs` | int | 5 | Liczba epok treningowych |
| `--lr` | float | 0.0002 | Współczynnik uczenia dla optymalizatorów |
| `--seed` | int | 999 | Ziarno losowości dla powtarzalności wyników |
| `--output_dir` | string | 'results' | Katalog do zapisywania wyników |
| `--use_kagglehub` | bool | False | Pobieranie zbioru danych z KaggleHub |
| `--max_images` | int | 200000 | Maksymalna liczba obrazów do użycia (tylko dla KaggleHub) |
| `--augment` | bool | True | Zastosowanie augmentacji danych |
| `--wgan` | bool | False | Użycie Wasserstein GAN zamiast standardowego GAN |

### Przykłady użycia

#### Trening z wykorzystaniem Wasserstein GAN:

```bash
python main.py --wgan --lr 0.00005 --num_epochs 20
```

#### Trening z augmentacją danych:

```bash
python main.py --augment --batch_size 32
```

#### Trening z pobieraniem danych z Kaggle i ograniczoną liczbą obrazów:

```bash
python main.py --use_kagglehub --max_images 10000 --output_dir results_10k
```

## Wyniki

Po ukończeniu treningu, w katalogu wyjściowym (domyślnie `results`) znajdziesz:

- `loss_plot.png` - wykres strat generatora i dyskryminatora
- `comparison.png` - porównanie prawdziwych i wygenerowanych obrazów
- `generator.pth` i `discriminator.pth` - zapisane modele
- `results_summary.txt` - podsumowanie treningu i metryki
- `individual_real/` - katalog z prawdziwymi obrazami
- `individual_fake/` - katalog z wygenerowanymi obrazami

## Częste problemy

### Problem: Błąd "No such file or directory" podczas zapisywania wyników

Katalog wyjściowy musi istnieć i mieć odpowiednie uprawnienia zapisu.

### Problem: CUDA out of memory

Zmniejsz rozmiar partii (batch size):

```bash
python main.py --batch_size 32
```

### Problem: Problemy z pobraniem danych z Kaggle

Upewnij się, że prawidłowo skonfigurowałeś API Kaggle i plik kaggle.json ma odpowiednie uprawnienia.

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Problem: Słabe wyniki

Spróbuj zwiększyć liczbę epok i użyć większej ilości danych:

```bash
python main.py --num_epochs 50 --use_kagglehub --max_images 100000
```
---