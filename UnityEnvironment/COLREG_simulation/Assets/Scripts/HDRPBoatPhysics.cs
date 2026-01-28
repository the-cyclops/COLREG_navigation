using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

[RequireComponent(typeof(Rigidbody))]
public class HDRPBoatPhysics : MonoBehaviour
{
    public WaterSurface waterSurface;
    
    [Header("Buoyancy Settings")]
    public float buoyancyStrength = 1.8f;    // Bilanciamento spinta/peso
    public float verticalDrag = 10.0f;       // Smorzamento oscillazioni (Damping)
    public float maxSubmergenceDepth = 0.12f; // Profondità per spinta max (80% altezza mesh)
    public Vector3 centerOfMassOffset = new Vector3(0, -0.1f, -0.1f); // Bilanciamento bow/stern

    [Header("Floater References")]
    public Transform[] floaters; 
    
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        // Sposta il baricentro per stabilità e livellamento
        rb.centerOfMass = centerOfMassOffset;
    }

    void FixedUpdate()
    {
        if (waterSurface == null || floaters.Length == 0) return;

        foreach (var floater in floaters)
        {
            ApplyPointBuoyancy(floater);
        }
    }

    void ApplyPointBuoyancy(Transform floater)
    {
        WaterSearchParameters search = new WaterSearchParameters();
        search.targetPositionWS = floater.position;

        if (waterSurface.ProjectPointOnWaterSurface(search, out WaterSearchResult result))
        {
            float waterHeight = result.projectedPositionWS.y;
            float depth = waterHeight - floater.position.y;

            if (depth > 0)
            {
                // 1. Calcolo Spinta Normalizzata
                float displacement = Mathf.Clamp01(depth / maxSubmergenceDepth);
                float weightPerPoint = (rb.mass * Mathf.Abs(Physics.gravity.y)) / floaters.Length;
                
                Vector3 buoyancyForce = Vector3.up * weightPerPoint * displacement * buoyancyStrength;

                // 2. Calcolo Drag Verticale (Damping)
                // Applichiamo forza contraria alla velocità verticale locale
                Vector3 velocityAtPoint = rb.GetPointVelocity(floater.position);
                Vector3 dampingForce = Vector3.up * -velocityAtPoint.y * verticalDrag * displacement;

                rb.AddForceAtPosition(buoyancyForce + dampingForce, floater.position, ForceMode.Force);
            }
        }
    }

    private void OnDrawGizmos()
    {
        if (rb != null)
        {
            // Centro di Massa in rosso
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(transform.TransformPoint(rb.centerOfMass), 0.05f);
            
            
        }
    }
}