using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class BoatAgent : Agent
{
    public HDRPBoatPhysics boatPhysics;
    private Rigidbody rb;
    public GameObject target;

    private float arenaSize = 20f;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public override void Initialize()
    {
        boatPhysics = GetComponent<HDRPBoatPhysics>();
        rb = GetComponent<Rigidbody>();

        initialPosition = Vector3.zero;
        initialRotation = Quaternion.identity;
    }

    void MoveTarget()
    {
        //TODO
    }

    public override void OnEpisodeBegin()
    {
        //Reset Boat Velocities
        boatPhysics.ResetVelocities();

        // Move Target 
        MoveTarget();

        // Reset Boat Position and Rotation
        transform.localPosition = initialPosition;
        transform.localRotation = initialRotation;

        Physics.SyncTransforms();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //TODO CHECK THE ORDER OF OBSERVATIONS - IT MATTERS?

        // Fetch Target Position Relative to Boat
        Vector3 targetRelativePos = transform.InverseTransformPoint(target.transform.position);
        sensor.AddObservation(targetRelativePos.normalized);

        // Fetch Target Distance
        float targetDistance = targetRelativePos.magnitude;
        // Normalized Distance (Assuming max distance of 20 units) TODO FIX 
        //sensor.AddObservation(targetDistance / arenaSize); 
        // Obs Index [3]: Target Distance - Rational normalization d/(d+k) with k=20. Range: [0, 1]
        sensor.AddObservation(targetDistance / (20f + targetDistance));

        // Fetch Boat Velocity
        // InverseTransformDirection converts world space velocity to local space
        // For example, if the boat is moving forward, the local velocity.z will be positive
        // If the boat is moving to the right, the local velocity.x will be positive and so on
        Vector3 localLinearVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        Vector3 normalizedLocalLinearVelocity = localLinearVelocity / boatPhysics.nominalMaxLinearSpeed;
        // Clamp magnitude to in -1 to 1 range
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalLinearVelocity, 1.0f));

        // Fetch Boat Angular Velocity
        Vector3 localAngularVelocity = transform.InverseTransformDirection(rb.angularVelocity);
        Vector3 normalizedLocalAngularVelocity = localAngularVelocity / boatPhysics.nominalMaxAngularSpeed;
        // Clamp magnitude to in -1 to 1 range
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalAngularVelocity, 1.0f));
        

        // Total Observations: 3 (targetRelativePos) + 1 (targetDistance) + 3 (linearVelocity) + 3 (angularVelocity) = 10
        // Adjust the Space Size in Vector Sensor in Behavior Parameters accordingly
    }

    // This method is called every step during training OR by your keyboard in Heuristic mode
    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions;
        float leftInput = Mathf.Clamp(continuousActions[0], -1f, 1f);
        float rightInput = Mathf.Clamp(continuousActions[1], -1f, 1f);

        // Apply forces based on jet inputs
        boatPhysics.SetJetInputs(leftInput, rightInput);

        // Calculate Reward
        //TODO

    }

    //public override void CalculateReward()
    //{
    //TODO
    //}

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle"))
        {   
            SetReward(-1.0f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            SetReward(1.0f);
            // 2. Coloriamo il pavimento di verde per feedback visivo (Opzionale ma bello)
            // StartCoroutine(SwapGroundMaterial(successMaterial, 0.5f));

            Debug.Log(GetCumulativeReward());
            EndEpisode(); 
        }
    }

    public void FixedUpdate() {
        // Penalità esistenziale per spronarlo a fare in fretta
        AddReward(-0.0005f); 
    }

    // This maps your keyboard to the Action Buffers
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Get reference to the continuous actions
        var continuousActions = actionsOut.ContinuousActions;
        
        // Vertical axis (W/S or Up/Down) -> Throttle
        float forwardInput = Input.GetAxis("Vertical"); 
        
        // Horizontal axis (A/D or Left/Right) -> Steering
        float turnInput = Input.GetAxis("Horizontal");


        //Mixing Commands for Jet Engines
        // By pressing W both jets push forward (+,+)
        // By pressing S both jets push backward (-,-)
        // By pressing A left jet backward, right jet forward (-,+)
        // By pressing D left jet forward, right jet backward (+,-)
        float leftJet = forwardInput + turnInput;
        float rightJet = forwardInput - turnInput;


        // Clamp (clip) the values between -1 and 1
        continuousActions[0] = Mathf.Clamp(leftJet, -1f, 1f);
        continuousActions[1] = Mathf.Clamp(rightJet, -1f, 1f);
    }

}