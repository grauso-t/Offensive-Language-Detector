# Dataset Card for Homophobia Detection Dataset (Twitter/X)

## Dataset Description

- **Paper:** TBC
- **Point of Contact:** Josh McGiff (Josh.McGiff@ul.ie)

## Dataset Summary

This dataset was developed to address the significant gap in online hate speech detection, particularly focusing on homophobia, which is often neglected in sentiment analysis research. It comprises tweets scraped from X (formerly Twitter), which have been labeled for the presence of homophobic content by volunteers from diverse backgrounds. This dataset is the largest open-source labelled English dataset for homophobia detection known to the authors and aims to enhance online safety and inclusivity.

## Supported Tasks

- **Task:** Homophobic hate speech detection.

## Languages

English.

## Dataset Structure

- **Data Fields:**
  - `tweet_text`: The text content of the tweet.
  - `label`: Binary label indicating the presence of homophobic content (0 = no homophobic content, 1 = homophobic content).
  - 'language': The language of the tweet, as tagged by X/Twitter.

## Dataset Creation

- **Curation Rationale:** The dataset was curated to enhance the detection and classification of homophobic content on social media platforms, particularly focusing on the gap where homophobia is underrepresented in current research.
- **Source Data:** Data was scraped from X (formerly Twitter) focusing on terms and accounts associated with the LGBTQIA+ community.
- **Annotation Process:** Annotations were made by three volunteers from different sexualities and gender identities using a majority vote for label assignment. Annotations were conducted in Microsoft Excel over several days.
- **Personal and Sensitive Information:** Usernames and other personal identifiers have been anonymized or removed. URLs have also been removed. The dataset contains sensitive content related to homophobia.

## Considerations for Using the Data

- **Social Impact:** The dataset is intended for research purposes to combat online hate speech and improve inclusivity and safety on digital platforms.
- **Ethical Considerations:** Given the sensitive nature of hate speech, researchers should consider the impact of their work on marginalised communities and ensure that their use of the dataset aims to reduce harm and promote inclusivity.
- **Legal and Privacy Concerns:** Researchers should comply with legal standards and ethical guidelines regarding hate speech and data privacy.

## Additional Information

- **License:** CC-BY-4.0

## References
```
@misc{mcgiff2024bridging,
    title={Bridging the Gap in Online Hate Speech Detection: A Comparative Analysis of BERT and Traditional Models for Homophobic Content Identification on X/Twitter},
    author={Josh McGiff and Nikolov N. S.},
    year={2024},
    journal={Applied and Computational Engineering},
    volume={64},
    pages={64-69},
    primaryClass={cs.CL}
}
```

## Acknowledgements

This work was conducted with the financial support of the Science Foundation Ireland Centre for Research Training in Artificial Intelligence under Grant No. 18/CRT/6223.
