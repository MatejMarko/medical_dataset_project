import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

questions = [
    "Are you self-employed?",
    "How many employees does your company or organization have?",
    "Does your employer provide mental health benefits as part of healthcare coverage?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
    "Do you feel that your employer takes mental health as seriously as physical health?",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?",
    "Do you know local or online resources to seek help for a mental health disorder?",
    "Do you believe your productivity is ever affected by a mental health issue?",
    "Do you have previous employers?",
    "Have your previous employers provided mental health benefits?",
    "Were you aware of the options for mental health care provided by your previous employers?",
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?",
    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?",
    "Do you think that discussing a physical health issue with previous employers would have negative consequences?",
    "Would you have been willing to discuss a mental health issue with your previous co-workers?",
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?",
    "Did you feel that your previous employers took mental health as seriously as physical health?",
    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?",
    "Would you bring up a mental health issue with a potential employer in an interview?",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?",
    "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?",
    "Do you have a family history of mental illness?",
    "Have you had a mental health disorder in the past?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "Do you work remotely?"
]


def diff_answers_column(column_of_values):
    unique = []
    for value in column_of_values:
        if value not in unique:
            unique.append(value)
    return unique


def columns_to_list(pandas_columns):
    list_with_column_names = []
    for c in pandas_columns.columns:
        list_with_column_names.append(c)
    return list_with_column_names


def list_to_dict(list_of_options):
    order = 1
    dictionary = {}
    for option in list_of_options:
        if option != "Nan":
            dictionary[order] = option
            order += 1
    return dictionary


def encoded(vect):
    enc = LabelEncoder()
    label_encoder = enc.fit(vect)
    t = label_encoder.transform(vect)
    return t


def int_encode_features(data_value, idx):
    return encoded(data_value[:, idx])


df = pd.read_csv("mental-heath-in-tech-2016.csv")
df.keys()
k = []
for i in range(1, len(df.keys()) + 1):
    k.append("Q" + str(i))
df.columns = k

for col in df:
    my_new_list = [s for s in df[col] if type(s) != str]
    if len(my_new_list) < len(df[col]):
        df[col] = df[col].replace([None], ["Nan"])

newDF = df.drop(
    ["Q3", "Q4", "Q10", "Q17", "Q19", "Q20", "Q21", "Q22", "Q24", "Q30", "Q38", "Q40", "Q43", "Q49", "Q50", "Q52",
     "Q53", "Q54", "Q55", "Q56", "Q57", "Q58", "Q59", "Q60", "Q61", "Q62"], axis=1)

target = newDF[["Q48"]].copy()
withoutTarget = newDF.drop(["Q48"], axis=1)

listOfDictsWithDifferentAnswers = []
index = 0
for column in withoutTarget:
    listOfDictsWithDifferentAnswers.append(list_to_dict(diff_answers_column(df[column])))
    index += 1

# change possible answers of 1. and 16. question, to make it more readable
listOfDictsWithDifferentAnswers[0][1] = "No"
listOfDictsWithDifferentAnswers[0][2] = "Yes"
listOfDictsWithDifferentAnswers[15][1] = "No"
listOfDictsWithDifferentAnswers[15][2] = "Yes"

list_of_columns = columns_to_list(withoutTarget)

numericalAnswers = []
for x in range(len(questions)):
    print(questions[x])
    print(listOfDictsWithDifferentAnswers[x])
    answer = input()
    while not answer.isdigit() or int(answer) < 1 or int(answer) > len(listOfDictsWithDifferentAnswers[x]):
        answer = input()
    numericalAnswers.append(answer)

userAnswers = []
for a in range(len(numericalAnswers)):
    if a == 0:
        if numericalAnswers[a] == "1":
            userAnswers.append(0)
        else:
            userAnswers.append(1)
    elif a == 15:
        if numericalAnswers[a] == "2":
            userAnswers.append(0)
        else:
            userAnswers.append(1)
    else:
        userAnswers.append(listOfDictsWithDifferentAnswers[a][int(numericalAnswers[a])])

userRow = pd.DataFrame([userAnswers], columns=list_of_columns)
completeWithout = withoutTarget.append(userRow, ignore_index=True)

userIllness = pd.DataFrame(["No"], columns=["Q48"])
completeTarget = target.append(userIllness, ignore_index=True)

data = completeWithout.values

for i in range(completeWithout.shape[1]):
    data[:, i] = int_encode_features(data, i)
data = data.astype(float)
target = completeTarget.values.ravel()

data_without_last = data[:-1, :]
target_without_last = target[:-1]
userInfo = data[-1, :].reshape(1, -1)

clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(data_without_last, target_without_last)  
prediction = clf.predict(userInfo)
  
if prediction[0] == 'Yes':
    print("Accordingly to our machine learning model you suffer from some kind of mental health disease. You should confirm this result with a doctor. Good luck :)")
elif prediction[0] == 'No':
    print("Accordingly to our machine learning model you don't suffer from any kind of mental health disease. However you should check it with a doctor.")
else:
    print("Our machine learning model couldn't determine your mental health state. You should consider consulting with a doctor.")
