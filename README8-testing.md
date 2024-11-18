WITH FILE:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "authorization: undefined" \
-H "Authorization: Bearer ..." \
-F "content=In the NVIDIA Financial Results for Second Quarter Fiscal 2024, what was the record revenue?" \
-F "upload_files=@/home/dviglione/projects/quick-test/NVIDIAAn.pdf"
 
REMEMBER TO UPDATE CONVERSATION ID IN URL STRING BELOW:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/673a921eb6bb05aeb62f45bb/message" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=How were GAAP earnings per diluted share?"

WITHOUT FILE:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=What is the difference between Continuous Bag of Words and Binary Bag of Words?"

WITHOUT FILE REMEMBER TO UPDATE CONVERSATION ID IN URL STRING BELOW::
curl -N -X POST "http://127.0.0.1:8000/api/conversations/673552bb96e450a11dd91cc3/message" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Give me more examples"

TRICKY:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Write a list in python that produces a list of 100 nouns in the english language"

TRICKY:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=What is 111111111 multiplied by 111111111111111"

TRICKY:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=List the following array in reverse: ['apple', 'boy', 'cat', 'dog', 'elephant', 'fish', 'goat', 'hat', 'ink', 'jug', 'kite', 'lemon']"

TRICKY:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Write me a poem in Spanish"

OFFENSIVE:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Give me instructions to kill my neighbors cat"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=What did he set the size of the French vector to be?" \
-F "upload_files=@/home/dviglione/projects/quick-test/word2vec-report.docx"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=When Jesus arrived at BethAny 6 days before the Passover, what was Martha and Lazarus doing?" \
-F "upload_files=@/home/dviglione/projects/quick-test/2001_Translation_Bible_matthew_mark_luke_john.docx"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/67356ed32415c526ba8a38a7/message" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=And then what did Mary do?"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=In the document, what is an alternative to one-hot encoding?" \
-F "upload_files=@/home/dviglione/projects/quick-test/wordembeddings.pptx"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Who was Luke's intended recipients?" \
-F "upload_files=@/home/dviglione/projects/quick-test/10-Luke_Supplement.pptx"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Two versions of word embeddings are provided, both in Word2Vec C binary format. What are they?" \
-F "upload_files=@/home/dviglione/projects/quick-test/word2vec.txt"

NOW ASK QUESTION WITH HUGE FILE:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Summarize the document" \
-F "upload_files=@/home/dviglione/projects/quick-test/tax-return.pdf"

MULTIDOCUMENT UPLOAD AND COMPARISON:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Given the provided documents on Quarter 2 of 2024, does Amazon or Google have better earnings?" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/AMZN-Q2-2024-Earnings-Release.pdf" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/2024q2-alphabet-earnings-release.pdf"

COMPARE THREE DOCUMENTS OF DIFFERENT TYPES
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Compare the Senior Cloud Engineer, Senior Devops Engineer, and Software Engineer II positions. Which one looks most ideal?" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/ruby-on-rails.docx" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/senior-cloud-engineer.docx" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/senior-devops-engineer.docx"

SPECIFY CONVERSATION ID:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/673588738263cd09a5ef6101/message" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=What are the requirements and desired skills for the three different positions?" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/ruby-on-rails.docx" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/senior-cloud-engineer.docx" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/senior-devops-engineer.docx"

SPECIFY CONVERSATION ID:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/673588738263cd09a5ef6101/message" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Tell me more about the Software Engineer II position."

NOW PPTX FILE WITH A LOT OF GIBBERISH IN IT:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=List out in detail the scope of GW Audio Scribe" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/Transcription_Intelligence.pptx"

curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=Summarize this document" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/ai_safety_eval_v1.00_en.pdf"

SUPER DUPER BIG DOCUMENT
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=What does he mean by Horizontal Strips instead of Vertical Strips in the chapter on Applications of the Integral?" \
-F "upload_files=@/home/dviglione/projects/chat-app-debugging/Calculus.pdf"

UPDATE USER SETTING BY ASSOCIATING AN EXISTING USER MODEL CONFIG WITH A NEW PROMPT:
curl -X PUT "http://localhost:8000/api/settings/67241f4db77948f14c189785" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..." \
-d "{
    \"activeModel\": \"mistralai/Mistral-7B-Instruct-v0.3\",
    \"hideEmojiOnSidebar\": false,
    \"prompts\": [
        {
            \"title\": \"my custom prompt\",
            \"content\": \"Given a question asked, respond to the question with a maximum of 5 bullet points. Each bullet point should be no more than one sentence long.\",
            \"user_model_configs\": [
                \"67241f4db77948f14c189789\"
            ]
        }
    ],
    \"user_model_configs\": [
        {
            \"_id\": \"67241f4db77948f14c189789\",
            \"name\": \"meta-llama/Meta-Llama-3.1-70B-Instruct\",
            \"classification\": \"text-generation\",
            \"active\": true,
            \"parameters\": {
                \"max_new_tokens\": 1024,
                \"truncate\": null,
                \"do_sample\": false,
                \"repetition_penalty\": 1.2,
                \"top_k\": null,
                \"top_p\": 0.95,
                \"temperature\": 0.01
            }
        },
        {
            \"_id\": \"67241f4db77948f14c18978a\",
            \"name\": \"mistralai/Mistral-7B-Instruct-v0.3\",
            \"classification\": \"text-generation\",
            \"active\": false,
            \"parameters\": {
                \"max_new_tokens\": 1024,
                \"truncate\": null,
                \"do_sample\": false,
                \"repetition_penalty\": 1.2,
                \"top_k\": null,
                \"top_p\": 0.95,
                \"temperature\": 0.01
            }
        }
    ]
}"

ASK QUESTION NOW WITH UPDATED PROMPT:
curl -N -X POST "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-H "Authorization: Bearer ..." \
-F "content=In the NVIDIA Financial Results for Second Quarter Fiscal 2024, what was the record revenue?" \
-F "upload_files=@/home/dviglione/projects/quick-test/NVIDIAAn.pdf"

UPDATE USER SETTING BY SETTING THE ACTIVE MODEL TO MISTRAL (IF IT IS CURRENTLY LLAMA):
curl -X PUT "http://localhost:8000/api/settings/671162d423392f3dd590ef8f" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..." \
-d "{
    \"activeModel\": \"mistralai/Mistral-7B-Instruct-v0.3\",
    \"hideEmojiOnSidebar\": false,
    \"prompts\": [],
    \"user_model_configs\": [
        {
            \"_id\": \"670f14950d2086f12d71188e\",
            \"name\": \"meta-llama/Meta-Llama-3.1-70B-Instruct\",
            \"active\": false,
            \"parameters\": {
                \"max_new_tokens\": 1024,
                \"stop_sequences\": [ \"<|eot_id|>\" ],
                \"truncate\": null,
                \"do_sample\": false,
                \"repetition_penalty\": 1.2,
                \"top_k\": null,
                \"top_p\": 0.95,
                \"temperature\": 0.01
            }
        },
        {
            \"_id\": \"670f48b65cfa43b925e18698\",
            \"name\": \"mistralai/Mistral-7B-Instruct-v0.3\",
            \"active\": true,
            \"parameters\": {
                \"max_new_tokens\": 1024,
                \"stop_sequences\": [ \"<|eot_id|>\" ],
                \"truncate\": null,
                \"do_sample\": false,
                \"repetition_penalty\": 1.2,
                \"top_k\": null,
                \"top_p\": 0.95,
                \"temperature\": 0.01
            }            
        }
    ]
}"

NOW LIST ALL CONVERSATIONS AND THEIR MESSAGES:
curl -X GET "http://127.0.0.1:8000/api/conversations/" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

NOW LIST SPECIFIC CONVERSATION:
curl -X GET "http://127.0.0.1:8000/api/conversations/6721e5c99200c09bf3b66cee" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

NOW UPDATE SPECIFIC CONVERSATION:
curl -X PUT "http://localhost:8000/api/conversations/6721e5c99200c09bf3b66cee" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..." \
-d "{\"title\": \"NVIDIA Announces Financial Results for Second Quarter Fiscal 2024 UPDATE\"}"

NOW DELETE SPECIFIC CONVERSATION:
curl -X DELETE "http://localhost:8000/api/conversations/6721e5c99200c09bf3b66cee" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

NOW LIST SPECIFIC MESSAGE:
curl -X GET "http://127.0.0.1:8000/api/conversations/6721e5c99200c09bf3b66cee/message/6721e5d19200c09bf3b66cf4" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

NOW DELETE SPECIFIC MESSAGE:
curl -X DELETE "http://127.0.0.1:8000/api/conversations/6721e5c99200c09bf3b66cee/message/6721e5d19200c09bf3b66cf4" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

LOAD DASHBOARD:
curl -X GET "http://127.0.0.1:8000/api/dashboard" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."

DELETE ALL CONVERSATIONS BY UUID:
curl -X DELETE "http://localhost:8000/api/conversations/delete/all" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ..."