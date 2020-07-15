from scipy.spatial.distance import cosine


class TestEncoder:
    def test_declutr_small(self, declutr_small):
        inputs = [
            "A smiling costumed woman is holding an umbrella.",
            "A happy woman in a fairy costume holds an umbrella.",
            "Two men are smiling and laughing at the cats playing on the floor.",
        ]
        embeddings = declutr_small(inputs, batch_size=3)
        assert cosine(embeddings[0], embeddings[1]) < cosine(embeddings[0], embeddings[2])
