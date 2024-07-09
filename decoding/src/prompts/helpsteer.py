# 9-shot code generation
code_prompt = '''
Explain the difference between deism and theism. Make sure to include at least 4 key characteristics of each.

Deism and theism are two different philosophical views that have different beliefs about the existence and nature of a deity.
Here are some key characteristics of each
Deism: - Believes in a distant and impersonal god who created the universe but does not interfere with it. 
Emphasizes reason and the natural world as the way to understand the divine.
Rejects religious traditions and institutions as unreliable sources of knowledge about God.
Often views the Bible as just one more piece of evidence for the existence of God, rather than a revelation from God.
Theism: - Believes in a personal and involved god who interacts with the world and with humans. 
- Emphasizes faith and religious experiences as the way to understand the divine.
- Accepts religious traditions and institutions as reliable sources of knowledge about God.
- Often views the Bible as a direct revelation from God.
In general, deism emphasizes reason and the natural world, while theism emphasizes faith and religious experiences.



What are some of the health benefits of eating beets? Can they help with blood pressure or cancer prevention?

Yes,beets are a good source of potassium, which can help regulate blood pressure.
Beets are also a good source of nitrates, which have been linked to improved athletic performance and reduced inflammation.





What does it mean to be unblinking and how is it used in a sentence?

The phrase "unblinking" typically refers to someone who is staring intently or not blinking.
It can be used in a sentence like this: "The soldier stood unblinking in the face of danger."
'''

# 5-shot (T/F)
# (https://www.mathplayground.com/wpdatabase/wpindex.html)
# (https://www.analyzemath.com/middle_school_math/grade_8/problems.html)
evaluate_prompt = '''
What are some of the health benefits of eating beets? Can they help with blood pressure or cancer prevention?

Yes,beets are a good source of potassium, which can help regulate blood pressure.
# Is the above:
# (A) Helpful
# (B) Unhelpful
# The above sentence is: (A)

Beets are also a good source of nitrates, which have been linked to improved athletic performance and reduced inflammation.
# Is the above:
# (A) Helpful
# (B) Unhelpful
# The above sentence is: (A)

Beets can help prevent cancer.
# Is the above:
# (A) Helpful
# (B) Unhelpful
# The above sentence is: (B)


What does it mean to be unblinking and how is it used in a sentence?

The phrase "unblinking" typically refers to someone who is staring intently or not blinking.
# Is the above:
# (A) Helpful
# (B) Unhelpful
# The above sentence is: (A)

It can be used in a sentence like this: "The soldier stood unblinking in the face of danger."
# Is the above:
# (A) Helpful
# (B) Unhelpful
# The above sentence is: (A)


'''

choice_prefix = ['# Is the above:', '# (A) Helpful', '# (B) Unhelpful', '# The above sentence is:']


