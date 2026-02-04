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

    private float arenaRadius = 15f;

    [SerializeField] GameObject[] obstacles;
    [SerializeField] GameObject intruderVessel1;
    [SerializeField] GameObject intruderVessel2;
    
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    public override void Initialize()
    {
        boatPhysics = GetComponent<HDRPBoatPhysics>();
        rb = GetComponent<Rigidbody>();

        initialPosition = Vector3.zero;
        initialRotation = Quaternion.identity;
    }

    private bool CheckTargetPosition(Vector2 pos)
    {
        float minDistance = 3.0f; // Minimum distance from obstacles

        foreach (GameObject obstacle in obstacles)
        {
            Vector3 obstaclePos = obstacle.transform.localPosition;
            float distance = Vector2.Distance(new Vector2(obstaclePos.x, obstaclePos.z), pos);
            if (distance < minDistance)
            {
                return false; // Too close to an obstacle
            }
        }
        return true; // Valid position
    }   

    // This method creates a new
    private void MoveTarget()
    {
        // Random.insideUnitCircle returns a random point inside a circle with radius 1.
        // We multiply it by (arenaRadius - 1) to ensure the target stays within the arena bounds, leaving a 1 unit margin from the edge. 

        Vector2 randomCircle = UnityEngine.Random.insideUnitCircle * (arenaRadius-1);
        while (!CheckTargetPosition(randomCircle))
        {
            randomCircle = UnityEngine.Random.insideUnitCircle * (arenaRadius-1);
        }
        Vector3 targetPosition = new Vector3(randomCircle.x, 0.0f, randomCircle.y);
        target.transform.localPosition = targetPosition;
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

        // --- SELF & TARGET OBSERVATIONS ---

        // Fetch Target Position Relative to Boat
        Vector3 targetRelativePos = transform.InverseTransformPoint(target.transform.position);
        sensor.AddObservation(targetRelativePos.normalized);

        // Fetch Target Distance
        float targetDistance = targetRelativePos.magnitude;
        // Normalized Distance (Assuming max distance of 20 units) TODO FIX 
        // Obs Index [3]: Target Distance - Rational normalization d/(d+k) with k=20. Range: [0, 1]
        sensor.AddObservation(targetDistance / (20f + targetDistance));

        // Fetch Boat Velocity
        // InverseTransformDirection converts world space velocity to local space
        Vector3 localLinearVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        Vector3 normalizedLocalLinearVelocity = localLinearVelocity / boatPhysics.nominalMaxLinearSpeed;
        // Clamp magnitude to in -1 to 1 range
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalLinearVelocity, 1.0f));

        // Fetch Boat Angular Velocity
        Vector3 localAngularVelocity = transform.InverseTransformDirection(rb.angularVelocity);
        Vector3 normalizedLocalAngularVelocity = localAngularVelocity / boatPhysics.nominalMaxAngularSpeed;
        // Clamp magnitude to in -1 to 1 range
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalAngularVelocity, 1.0f));

        // --- COLREG / INTRUDER VESSEL OBSERVATIONS ---
        // Necessary for calculating CPA and Velocity Obstacles in Python
    
        // INTRUDER VESSEL 1
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel1 != null && intruderVessel1.activeInHierarchy)
        {
            // 1. Relative Position of the first intruder vessel (Local space)
            Vector3 intruder1RelativePos = transform.InverseTransformPoint(intruderVessel1.transform.position);
            sensor.AddObservation(intruder1RelativePos.normalized); 
        
            float intruder1Dist = intruder1RelativePos.magnitude;
            sensor.AddObservation(intruder1Dist / (20f + intruder1Dist)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            // We calculate the vector difference in world space, then convert to local
            Vector3 relativeVelocityWorld1 = intruderVessel1.GetComponent<Rigidbody>().linearVelocity - rb.linearVelocity;
            Vector3 localRelativeVelocity1 = transform.InverseTransformDirection(relativeVelocityWorld1);
        
            // Normalize by specific factor (e.g., sum of max speeds or just max speed)
            sensor.AddObservation(Vector3.ClampMagnitude(localRelativeVelocity1 / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            sensor.AddObservation(Vector3.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector3.zero); // Rel Vel
        }

        // INTRUDER VESSEL 2
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel2 != null && intruderVessel2.activeInHierarchy)
        {
            // 1. Relative Position of the second intruder vessel (Local space)
            Vector3 intruder2RelativePos = transform.InverseTransformPoint(intruderVessel2.transform.position);
            sensor.AddObservation(intruder2RelativePos.normalized); 
        
            float intruder2Dist = intruder2RelativePos.magnitude;
            sensor.AddObservation(intruder2Dist / (20f + intruder2Dist)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            // We calculate the vector difference in world space, then convert to local
            Vector3 relativeVelocityWorld2 = intruderVessel2.GetComponent<Rigidbody>().linearVelocity - rb.linearVelocity;
            Vector3 localRelativeVelocity2 = transform.InverseTransformDirection(relativeVelocityWorld2);
        
            // Normalize by specific factor (e.g., sum of max speeds or just max speed)
            sensor.AddObservation(Vector3.ClampMagnitude(localRelativeVelocity2 / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            sensor.AddObservation(Vector3.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector3.zero); // Rel Vel
        }

    // Total Observations: 3 (targetRelativePos) + 1 (targetDistance) + 3 (linearVelocity) + 3 (angularVelocity) + 3 (intruder1RelativePos) + 1 (intruder1Dist) + 3 (intruder1RelVel) + 3 (intruder2RelativePos) + 1 (intruder2Dist) + 3 (intruder2RelVel) = 24
    // // Adjust the Space Size in Vector Sensor in Behavior Parameters accordingly 
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
        } else 
        if (collision.gameObject.CompareTag("Wall")){
            SetReward(-2f);
            Debug.Log(GetCumulativeReward());
            EndEpisode();
        } else
        if (collision.gameObject.CompareTag("Boat")){
            SetReward(-1.5f);
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