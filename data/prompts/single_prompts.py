from langchain import PromptTemplate

string_templates = {

    'age':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are a {age} years old person. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'sex':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are a {sex}. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'ethnicity':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you have a {ethnicity} ethnic background. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'country_of_birth':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you were born in {country_of_birth}. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'country_of_residence':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are a resident in {country_of_residence}. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'nationality':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are a national of {nationality}. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'is_student':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are {is_student} student. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
    ,
    'employment_mode':
        """
        Irony is a literary or rhetorical device in which there is a discrepancy between what is said and what is meant,\n 
        or between expectations and reality. It often involves a contrast between appearance and reality,\n 
        or a situation in which the outcome is the opposite of what one might expect.\n
        Imagine that you are {employment_mode} employed. Following is a message and its' reply from a conversation.\n 
        Message : {message}\n
        Reply   : {reply}\n
        Do you find the above reply ironic? Answer Yes/No\n
        """
}

single_mapping = {
    'age': "Age",
    'sex': "Sex",
    'ethnicity': "Ethnicity simplified",
    'country_of_birth': 'Country of birth',
    'country_of_residence': 'Country of residence',
    'nationality': 'Nationality',
    'is_student': 'Student status',
    'employment_mode': 'Employment status'
}

prompt_templates = {
    'age': PromptTemplate(template=string_templates['age'], input_variables=['age', 'message', 'reply']),
    'sex': PromptTemplate(template=string_templates['sex'], input_variables=['sex', 'message', 'reply']),
    'ethnicity': PromptTemplate(template=string_templates['ethnicity'],
                                input_variables=['ethnicity', 'message', 'reply']),
    'country_of_birth': PromptTemplate(template=string_templates['country_of_birth'],
                                       input_variables=['country_of_birth', 'message', 'reply']),
    'country_of_residence': PromptTemplate(template=string_templates['country_of_residence'],
                                           input_variables=['country_of_residence', 'message', 'reply']),
    'nationality': PromptTemplate(template=string_templates['nationality'],
                                  input_variables=['nationality', 'message', 'reply']),
    'is_student': PromptTemplate(template=string_templates['is_student'],
                                 input_variables=['is_student', 'message', 'reply']),
    'employment_mode': PromptTemplate(template=string_templates['employment_mode'],
                                      input_variables=['employment_mode', 'message', 'reply'])
}
