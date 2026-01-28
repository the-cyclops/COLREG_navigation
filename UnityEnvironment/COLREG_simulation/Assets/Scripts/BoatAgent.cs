using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class BoatAgent : Agent
{
    public HDRPBoatPhysics boatPhysics;

    public GameObject target;

    private float arenaSize = 20f;

    public override void Initialize()
    {
        boatPhysics = GetComponent<HDRPBoatPhysics>();
    }

    public override void OnEpisodeBegin()
    {
        //Reset Boat Velocities
        boatPhysics.ResetVelocities();

        // Move Target 
        // TODO
        // Reset Boat Position and Rotation
        // TODO 
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //TODO CHECK THE ORDER OF OBSERVATIONS - IT MATTERS?

        // Fetch Target Direction
        Vector3 directionToTarget = (target.transform.position - transform.position).normalized;
        sensor.AddObservation(directionToTarget);

        // Fetch Target Alignment  [-1 to 1] (1 = facing target, -1 = facing away, 0 = perpendicular)
        float targetAlignment = Vector3.Dot(transform.forward, directionToTarget);
        sensor.AddObservation(targetAlignment);

        // Fetch Target Distance
        float targetDistance = Vector3.Distance(transform.position, target.transform.position);
        sensor.AddObservation(targetDistance / arenaSize); 
        // Normalized Distance (Assuming max distance of 20 units) TODO FIX 

        Rigidbody rb = boatPhysics.GetComponent<Rigidbody>();

        // Fetch Boat Velocity
        // InverseTransformDirection converts world space velocity to local space
        // For example, if the boat is moving forward, the local velocity.z will be positive
        // If the boat is moving to the right, the local velocity.x will be positive and so on
        Vector3 localVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        sensor.AddObservation(localVelocity / boatPhysics.maxSpeed);

        // Fetch Boat Angular Velocity
        Vector3 localAngularVelocity = rb.angularVelocity;
        // In this case, we consider only the y component for yaw rotation
        sensor.AddObservation(localAngularVelocity.y / boatPhysics.maxAngularSpeed); // Normalized Angular
        

        // Total Observations: 3 (direction) + 1 (alignment) + 1 (distance) + 3 (velocity) + 1 (angular velocity) = 9
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

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            //TODO GIVE REWARD
            //SetReward(1.0f);

            // 2. Coloriamo il pavimento di verde per feedback visivo (Opzionale ma bello)
            // StartCoroutine(SwapGroundMaterial(successMaterial, 0.5f));

            EndEpisode(); 
        }
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
        continuousActions[0] = leftJet;
        continuousActions[1] = rightJet;
    }

}