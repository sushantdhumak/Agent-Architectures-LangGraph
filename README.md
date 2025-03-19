# Agent-Architectures-LangGraph

AI Agent Architectures is the complex structures that shape how machines perceive, reason, and act in their environments in the pursuit of autonomous intelligence.

---

### Basic Reflection (reflection_agent.py)

This simple example composes two LLM calls: a generator and a reflector. 

The generator tries to respond directly to the user's requests. 

The reflector is prompted to role play as a teacher and offer constructive criticism for the initial response.

The loop proceeds a fixed number of times, and the final generated output is returned.

![image](https://github.com/user-attachments/assets/79a3f193-346d-46a8-8579-c1dda7181212)

---

### Reflexion (reflexion_agent.py)

`Reflexion` is an architecture designed to learn through verbal feedback and self-reflection. 

Within Reflexion, the actor agent provides explicit critiques for each response, grounding its criticism in external data.

It is required to generate citations and clearly identify both the superfluous elements and missing aspects of the generated content.

This process makes the reflections more constructive, enabling the agent to better adjust its responses based on the feedback provided.

![image](https://github.com/user-attachments/assets/d8ebb508-b739-4489-884d-1049b9a9b7e0)

---
