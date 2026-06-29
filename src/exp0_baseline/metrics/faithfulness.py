from .assertions import get_assertions
from .decisiveness import get_decisiveness
from .uncertainty import get_uncertainty

def get_faithfulness(answer, sampled_answers, confidence_score=None):

    ### Get Assertions
    assertions = get_assertions(answer)
    num_assertions = len(assertions)

    ### Compute Confidence & Faithfulness Summand Per Assertion
    total_uncertainty_gap = 0
    conf_scores = []
    dec_scores = []
    conf_responses_all = []

    for idx, assertion in enumerate(assertions):

        if assertion != "": # Skip Punted Questions

            ### Score Decisiveness
            text = assertion.replace("\n\n", " ").replace("\n", " ")
            decisiveness_score = get_decisiveness(answer=str(answer).replace("\n\n", " ").replace("\n", " "))
            # decisiveness_score = get_decisiveness(answer=text)    # diff. from MF

            if decisiveness_score==-1: # If decisiveness score failed, skip
                print("Decisiveness score -1; skipping...")
                continue

            ### Score Confidence 
            if confidence_score!=None:
                conf_score = confidence_score

                if conf_score == -1: # If confidence score failed, skip
                    print("Confidence score -1; skipping...")
                    continue

                dec_scores.append(decisiveness_score)
                conf_scores.append(conf_score)
                total_uncertainty_gap += abs(decisiveness_score - conf_score)
            
            else:
                conf_score, conf_responses = get_uncertainty( 
                    assertion=text,  
                    sampled_answers=sampled_answers, 
                )

                if conf_score == -1: # If confidence score failed, skip
                    print("Confidence score -1; skipping...")
                    continue
                dec_scores.append(decisiveness_score)
                conf_scores.append(conf_score)
                total_uncertainty_gap += abs(decisiveness_score - conf_score)
                conf_responses_all += conf_responses
    
        else: # Skip if Assertion Blank
            print("Assertion blank; skipping...")
            continue

    # If Confidence & Decisivenss Scoring Failed
    if len(conf_scores)==0:  
        print("No valid assertions or confidence scores; skipping...")
        return -1, -1, -1, assertions, []
    
    ### Compute Faithfulness Score & Average Response-Level Confidence   
    faithfulness_score = 1. - (1. / num_assertions) * total_uncertainty_gap 
    avg_conf_score = 1. * sum(conf_scores) / len(conf_scores)
    avg_dec_score = 1. * sum(dec_scores) / len(dec_scores)

    return faithfulness_score, avg_conf_score, avg_dec_score, assertions, conf_responses_all