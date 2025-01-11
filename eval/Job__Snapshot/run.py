from promptflow.client import PFClient
from azure.identity import DefaultAzureCredential
import os

def main():

    pf = PFClient(
        credential=DefaultAzureCredential(),
        subscription_id="e0fd569c-e34a-4249-8c24-e8d723c7f054",
        resource_group_name="rg-qunsongai",
    )

    os.environ['AZUREML_ARM_SUBSCRIPTION'] = "e0fd569c-e34a-4249-8c24-e8d723c7f054"
    os.environ['AZUREML_ARM_RESOURCEGROUP'] = "rg-qunsongai"
    os.environ['AZUREML_ARM_WORKSPACE_NAME'] = "qunsong-0951"

    use_qr = "true"
    metrics = "gpt_groundedness,f1_score,gpt_fluency,gpt_coherence,gpt_similarity,gpt_relevance,violence,self_harm,sexual,protected_material,xpia,hate_unfairness,bleu_score,gleu_score,meteor_score,rouge_score,eci"

    if use_qr == "true":
        data = "/Users/matsu/Library/CloudStorage/OneDrive-Microsoft/dev/RAG-on-Azure-Hands-on-Lab-1/eval/Job__Snapshot/samples_qr.json"

        column_mapping = {
            "ground_truth": "${data.ground_truth}",
            "context": "${data.context}",
            "groundedness_service_flight": "true",
            "metrics": metrics,
            "use_qr": use_qr,
            "query": "${data.query}",
            "response": "${data.response}",
        }
    else:
        data = "/Users/matsu/Library/CloudStorage/OneDrive-Microsoft/dev/RAG-on-Azure-Hands-on-Lab-1/eval/Job__Snapshot/samples_qr.json"

        column_mapping = {
            "ground_truth": "${data.ground_truth}",
            "question": "${data.question}",
            "answer": "${data.answer}",
            "context": "${data.context}",
            "groundedness_service_flight": "true",
            "metrics": metrics,
            "use_qr": use_qr,
        }

    # create run with the flow function and data
    run = pf.run(
        flow="/Users/matsu/Library/CloudStorage/OneDrive-Microsoft/dev/RAG-on-Azure-Hands-on-Lab-1/eval/Job__Snapshot/flow.dag.yaml",
        data=data,
        column_mapping=column_mapping,
        stream=False,
    )

    df = pf.get_details(run)
    print(df.head(10))
    print(pf.get_metrics(run))

# Call the aggregate_results function and store the result
    output = aggregate_results(results, selected_metrics, thresholds)

    # Print the final output
    print("Aggregate Output:")
    for key, value in output.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
