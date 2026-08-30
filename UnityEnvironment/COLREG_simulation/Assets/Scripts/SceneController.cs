using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;

public class SceneController : MonoBehaviour
{
    void Start()
    {
        float isEval = Academy.Instance.EnvironmentParameters.GetWithDefault("is_eval_scene", 0f);
        
        if (isEval > 0.5f && SceneManager.GetActiveScene().name != "NewEvalScene")
        {
            SceneManager.LoadScene("NewEvalScene");
        }
    }
}