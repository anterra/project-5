### Project 5 Proposal

# Vinyasa Krama: Deep Learning Generated Yoga Classes

### Anterra Kennedy

## Proposal:

I propose the invention of a data-driven yoga application which creates safe, completely unique, and compelling yoga classes which can be taken at the click of a button. Especially given COVID-19, it will benefit people stuck at home greatly to have new yoga classes to take any time.

### Background

The rise in popularity of yoga in the United States has turned it into a \$12 billion a year industry, and yet it remains the only industry without any kind of official licensure or credentialing system. The introduction of yoga to the West and its subsequent commercialization has created a culture of consumption surrounding yoga; offered so plentifully and consumed so rapidly that it is often not sat with or reflected upon, teachers in meeting that demand have to come up with so many new classes each week that they aren’t practicing those sequences on their own. Indeed, their own teacher training likely took place over only a couple of weekends, meaning the teachings have not been deeply absorbed or embodied — some chain studios have teachers just memorize a script — and the classes they’re offering consequently may not have safe, efficient or well-rounded sequencing.

As a lifelong practitioner of yoga, I have identified a real need for an injection of authenticity and informed, safe sequencing into the Western yoga community. Can data science replicate the firmly grounded and informed felt sense of a good yoga sequence that the likes of T. Krishnamacharya, the father of modern yoga, originally taught? The phrase "Vinyasa Krama" — the origin of the now popular ‘Vinyasa’ yoga — means a correctly organized sequence which progresses wisely. This ‘correct’ sequencing should include physical and biomechanical considerations for safety of the joints, necessary countermovements to each pose practiced, the energetic effect of the chosen asanas, and the overall arc and flow of the class. I want to use machine learing and neural networks to offer such authentic and well-rounded sequences to modern yoga teachers and practitioners.

### Goal

I propose an application which synthesizes thousands of available pre-made yoga sequences by real teachers, and includes further considerations for safety mechanics from other sources, to generate data-driven class sequences via a bi-directional LSTM neural network. This I will use to generate safe, compelling and unique sequences for Western yoga practitioners that are rooted in the ancient, time-tested principles of sequencing of Eastern Yogic philosophy.

The application could be used both by yoga teachers, to create classes for them to then teach in person to students, as well as by general yoga practitioners, who will be able to practice along with the animated front end of the application.

## Methodologies:

Use a bi-directional LSTM neural network to generate yoga classes.

- Input yoga classes by real teachers into the neural network, including the tens of thousands of yoga classes I scraped for Project 3, plus new class data all the time as I will establish a system to scrape the newly uploaded classes each day and store them in a SQL database that will be continuously queried.
- Utilize NLP pre-processing techniques of tokenization to isolate each pose and TF-IDF to remove poses in the classes that are very rare. This way the NN will have only a smaller subset of more common poses to choose from to build its classes.
- Make use of the wolfram language documentation, which contains every yoga pose, plus its relevant activated muscle groups, body position, benefits, contraindications, difficulty, and requisite poses that should precede and follow. This information will be given weight such that the NN will not build a class at random, but rather following these rules.
- I'm choosing bi-directional LSTM for the purpose of how I want the app to work: Typically in a yoga class there is a "peak posture", the most difficult pose at the peak of the class (about 2/3 of the way through or so). My app will allow the user to select the peak posture they would like to get to, and the app will build out in both directions from there, to create the right sequence of poses to build up to and cool down from the specified peak.
- Utilize the Unity Game Engine to create a 3D person with a logically structured skeleton to demonstrate each pose in the generated class. It will be animated so the user can follow along.
- Record voice overs calling out each pose for the app as well.

## Deliverables:

- Vinyasa Krama App: Complete application which generates unique yoga classes built around a peak pose of the user's choice, and allows them to either print out the list of poses (if they're a teacher), or follow along with a video demonstration with audio of the pose names.
- Code
- Presentation Slides
- Recording of Presentation

## Project Team:

- [Anterra Kennedy](https://linkedin.com/in/anterrakennedy)

## Technologies Used (Anticipated):

- Bi-Directional LSTM
- Wolfram Client Library
- Tensorflow/Keras
- GCP
- SQL
- Unity
- Jupyter Notebook
- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Flask
- HTML/CSS

## Skills Demonstrated

- Neural Networks
- Unsupvised Machine Learning
- Web Scraping
- Word Embeddings
