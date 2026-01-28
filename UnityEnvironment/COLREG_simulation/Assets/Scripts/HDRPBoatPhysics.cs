using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

[RequireComponent(typeof(Rigidbody))]
public class HDRPBoatPhysics : MonoBehaviour
{
    public WaterSurface waterSurface;
    
    [Header("Buoyancy Settings")]
    public float buoyancyStrength = 1.5f; 
    public float verticalDrag = 2.0f; // Il "Vertical Damping" richiesto
    public Vector3 centerOfMassOffset = new Vector3(0, -0.08f, 0);

    [Header("Floater References")]
    // Trascina qui i 6 Empty GameObjects posizionati con il vertex snapping
    public Transform[] floaters; 
    
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        // Fondamentale per la stabilità: sposta il peso verso il basso (le chiglie)
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
                // 1. Calcolo Forza di Galleggiamento (Buoyancy)
                // Dividiamo la massa per il numero di punti per distribuire il peso correttamente
                float weightPerPoint = (rb.mass * Mathf.Abs(Physics.gravity.y)) / floaters.Length;
                Vector3 buoyancyForce = Vector3.up * weightPerPoint * depth * buoyancyStrength;

                // 2. Calcolo Drag Verticale (Damping)
                // Applichiamo una forza contraria alla velocità verticale nel punto specifico
                Vector3 velocityAtPoint = rb.GetPointVelocity(floater.position);
                Vector3 dampingForce = Vector3.down * velocityAtPoint.y * verticalDrag;

                // 3. Applicazione delle forze al punto specifico
                rb.AddForceAtPosition(buoyancyForce + dampingForce, floater.position, ForceMode.Force);
            }
        }
    }

    // Visualizzazione nel Gizmos per debug
    private void OnDrawGizmos()
    {
        if (rb != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(transform.TransformPoint(rb.centerOfMass), 0.03f);
        }
    }
}