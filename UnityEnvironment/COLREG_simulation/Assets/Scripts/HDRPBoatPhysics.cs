using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class HDRPBoatPhysics : MonoBehaviour
{
    public WaterSurface waterSurface; 
    public float buoyancyStrength = 10f; 
    
    private Rigidbody rb;

    void Start() => rb = GetComponent<Rigidbody>();

    void FixedUpdate()
    {
        if (waterSurface == null) return;

        // In Unity 6, the property is 'targetPositionWS'
        WaterSearchParameters search = new WaterSearchParameters();
        search.targetPositionWS = transform.position; 

        if (waterSurface.ProjectPointOnWaterSurface(search, out WaterSearchResult result))
        {
            // In Unity 6, the property is 'projectedPositionWS'
            float waterHeight = result.projectedPositionWS.y; 
            float boatHeight = transform.position.y;

            if (boatHeight < waterHeight)
            {
                float depth = waterHeight - boatHeight;
                // Force = Gravity * Mass * Depth * Strength
                rb.AddForce(Vector3.up * Mathf.Abs(Physics.gravity.y) * rb.mass * depth * buoyancyStrength, ForceMode.Force);
            }
        }
    }
}