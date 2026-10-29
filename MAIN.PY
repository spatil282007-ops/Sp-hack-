
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the dataset
try:
    books = pd.read_csv("books.csv")
except FileNotFoundError:
    print("Error: books.csv file not found.")
    print("Keep books.csv and main.py in the same folder.")
    raise SystemExit

# Step 2: Validate and clean the data
required_columns = [
    "Title", "Author", "Genre", "Description", "Rating"
]

if not all(column in books.columns for column in required_columns):
    print("Error: CSV is missing required columns.")
    raise SystemExit

books = books.dropna(subset=[
    "Title", "Author", "Genre", "Description"
]).copy()

books["Title"] = books["Title"].astype(str).str.strip()
books = books[books["Title"] != ""]

books["Rating"] = pd.to_numeric(
    books["Rating"], errors="coerce"
).fillna(0)

books = books.reset_index(drop=True)

if books.empty:
    print("No valid books found in the dataset.")
    raise SystemExit

# Step 3: Combine genre and description
books["Features"] = (
    books["Genre"].astype(str) + " " +
    books["Description"].astype(str)
)

# Step 4: Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["Features"])

# Step 5: Calculate similarity between every pair of books
similarity_matrix = cosine_similarity(tfidf_matrix)


# Step 6: Create recommendation function
def recommend_books(book_name, number=5):
    matches = books[
        books["Title"].str.casefold()
        == book_name.strip().casefold()
    ]

    if matches.empty:
        print("\nBook not found!")
        print("Please enter a title from the available list.")
        return

    book_index = matches.index[0]

    # Get similarity scores for the selected book
    scores = list(enumerate(similarity_matrix[book_index]))

    # Sort books from most similar to least similar
    scores.sort(key=lambda item: item[1], reverse=True)

    print("\n" + "=" * 55)
    print("RECOMMENDED BOOKS")
    print("=" * 55)
    print("Selected book:", books.loc[book_index, "Title"])

    count = 0

    for index, score in scores:
        # Do not recommend the same book
        if index == book_index:
            continue

        book = books.iloc[index]

        print(f"\n{count + 1}. {book['Title']}")
        print(f"   Author: {book['Author']}")
        print(f"   Genre: {book['Genre']}")
        print(f"   Rating: {book['Rating']}/5")
        print(f"   Similarity: {score * 100:.2f}%")

        count += 1

        if count >= number:
            break

    if count == 0:
        print("No other books are available.")


# Step 7: Main program
def main():
    print("=" * 55)
    print("       BOOK RECOMMENDATION SYSTEM")
    print("=" * 55)
    print(f"\nTotal books available: {len(books)}")

    while True:
        print("\nAvailable books:")

        for title in books["Title"]:
            print(" -", title)

        choice = input(
            "\nEnter a book title or type 'exit': "
        ).strip()

        if choice.casefold() == "exit":
            print("\nThank you for using the system!")
            break

        if not choice:
            print("Please enter a book title.")
            continue

        recommend_books(choice)


if __name__ == "__main__":
    main()
