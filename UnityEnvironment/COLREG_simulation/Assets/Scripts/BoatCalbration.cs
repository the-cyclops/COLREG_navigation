using UnityEngine;

public class BoatCalibration : MonoBehaviour
{
    private HDRPBoatPhysics boatPhysics;
    private Rigidbody rb;

    [Header("Results (Watch these in Inspector)")]
    public float measuredMaxLinearSpeed = 0f;
    public float measuredMaxAngularSpeed = 0f;

    void Start()
    {
        boatPhysics = GetComponent<HDRPBoatPhysics>();
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate() // Use FixedUpdate for physics measurements
    {
        // Press 'T' for Linear Test (Both jets full forward)
        if (Input.GetKey(KeyCode.T))
        {
            boatPhysics.SetJetInputs(1f, 1f);
            if (rb.linearVelocity.magnitude > measuredMaxLinearSpeed)
                measuredMaxLinearSpeed = rb.linearVelocity.magnitude;
        }

        // Press 'R' for Angular Test (Left back, Right forward)
        if (Input.GetKey(KeyCode.R))
        {
            boatPhysics.SetJetInputs(-1f, 1f);
            if (Mathf.Abs(rb.angularVelocity.y) > measuredMaxAngularSpeed)
                measuredMaxAngularSpeed = Mathf.Abs(rb.angularVelocity.y);
        }
    }
}