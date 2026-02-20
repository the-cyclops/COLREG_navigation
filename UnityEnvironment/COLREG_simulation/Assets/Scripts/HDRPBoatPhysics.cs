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

    [Header("Stability")]
    public float sideDrag = 5.0f; // Resistenza laterale per evitare l'effetto saponetta

    [Header("Floater References")]
    public Transform[] floaters;

    public Transform leftJet;

    public Transform rightJet;

    public float maxThrust = 25f;

    // Approximate maximum linear and angular speeds, slightly higher to account for wave influence and prevent clamping
    public float nominalMaxLinearSpeed = 2.5f; 
    public float nominalMaxAngularSpeed = 1.4f;
    
    private Rigidbody rb;
    private float currentLeftInput = 0f;
    private float currentRightInput = 0f;

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

        Vector3 localVel = transform.InverseTransformDirection(rb.linearVelocity);

        // Applica una forza contraria SOLO all'asse X (laterale)
        // L'asse Z (avanti) e Y (verticale) vengono ignorati da questa forza
        rb.AddRelativeForce(new Vector3(-localVel.x * sideDrag, 0, 0), ForceMode.Acceleration);

        float leftForce = currentLeftInput * maxThrust;
        float rightForce = currentRightInput * maxThrust;

        rb.AddForceAtPosition(transform.forward * leftForce, leftJet.position);
        rb.AddForceAtPosition(transform.forward * rightForce, rightJet.position);
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

    public void ResetVelocities()
    {
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
        currentLeftInput = 0f;
        currentRightInput = 0f;
    }

    public void SetJetInputs(float leftInput, float rightInput)
    {
        currentLeftInput = leftInput;
        currentRightInput = rightInput;
    }
}