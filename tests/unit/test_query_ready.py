"""Confirmed availability uses real core/progress with byte-address SDK fakes."""
import time
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl, FakeNixlAgent, FakePrimaryPinning, _new_kvcr, _router_hint,
)
from test_guard_only_sources import _source
from kvcr.config import RemoteFWDramOptions
from kvcr.types import CacheTier, QueryStatus
from kvcr.ready_query import ReadyQueries


def pair():
    source, agent, source_control, pins, memory = _source()
    source_control.endpoint='tcp://source:1'
    target_control=FakeBytesControl('tcp://target:1')
    target=_new_kvcr(FakeNixlAgent(),FakePrimaryPinning(),target_control,
        key_adapter=SimpleNamespace(decode=lambda key:int.from_bytes(key,'little')),
        remote_options=RemoteFWDramOptions(guard_only_sources=True,eager_ctrl_connect=False))
    target.submit_hint(_router_hint(source_control.endpoint,[97,98]),'req')
    return source,target,source_control,target_control,agent,pins,memory


def exchange(left,right):
    for origin,destination in ((left,right),(right,left)):
        sent=list(origin.sent);del origin.sent[:len(sent)]
        destination.incoming.extend(payload for _,payload in sent)


def confirmed(target,left,right,keys=(b'a',b'b'),timeout_ms=5000):
    until=time.monotonic()+2
    while time.monotonic()<until:
        exchange(left,right)
        result=target.query_ready(keys,'req',timeout_ms=timeout_ms)
        if all(status is not QueryStatus.FETCHING for status,_ in result):return result
        time.sleep(.001)
    raise AssertionError('READY observation never completed')


def test_hint_is_authorization_not_confirmed_availability():
    source,target,left,right,agent,pins,memory=pair()
    assert target.query([b'b'],'req')==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]
    assert target.query_ready([b'b'],'req')==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]
    assert confirmed(target,left,right)==[
        (QueryStatus.FETCHABLE,CacheTier.REMOTE_G2),(QueryStatus.MISS,None)]
    # The opt-in did not change advisory legacy queries, even after a READY miss.
    assert target.query([b'b'],'req')==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]


def test_probe_does_not_need_source_caller_or_take_any_native_ownership():
    source,target,left,right,agent,pins,memory=pair()
    before=len(agent.xfers)
    with source._core._state_lock:
        assert confirmed(target,left,right)[0]==(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)
        assert source._core._local_dram._export_readers()==0
        assert not pins.searches and not pins.unpins
        assert len(agent.xfers)==before
        assert not source._core._remote_fw_dram._dangling_ops.source_writes
        assert not source._core._remote_fw_dram._remote_agents_by_target
    observation=target.query_ready_observations([b'a'],'req')[0]
    assert observation['valid_until_monotonic']<=time.monotonic()+5
    assert observation['source_incarnation']==source._core._remote_fw_dram._dangling_ops.incarnation
    assert observation['residency_generation'] is not None


def helper():
    clock=[100.0];messages=[]
    dangling=SimpleNamespace(sources={'tcp://source:1':'inc'},
        source_progress_epochs={('tcp://source:1','inc'):0})
    def remember(endpoint,payload):
        dangling.source_progress_epochs[endpoint,payload['sender_incarnation']]=payload['sender_progress_epoch']
    dangling.remember_source_progress=remember
    backend=SimpleNamespace(_kvcr=SimpleNamespace(_clock=lambda:clock[0],pool_layouts=[('',16)]),
        _dangling_ops=dangling,_metadata_retry_after={},
        _send_control=lambda p,endpoint,message:messages.append((endpoint,message)) or True)
    return ReadyQueries(backend),backend,clock,messages,SimpleNamespace(_stop_requested=False)


def response(message,**changes):
    value=dict(type='ready_probe_reply',probe_id=message['probe_id'],
        request_generation=message['request_generation'],keys=list(message['keys']),
        deadline=message['deadline'],sender_control_endpoint='tcp://source:1',
        sender_incarnation='inc',requested_epoch=message['source_progress_epoch'],
        requested_incarnation=message['source_incarnation'],
        sender_progress_epoch=0,state='ready',ready=[True]*len(message['keys']),
        generations=[1]*len(message['keys']),export_sequence=2)
    value.update(changes)
    return value


def test_cache_uses_target_clock_five_seconds_and_reprobes_after_expiry():
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
    q.receive(response(messages[-1][1]));q.poll(p)
    observation=q.observations((b'a',),'r')[0]
    assert observation['observed_at_monotonic']==100
    assert observation['valid_until_monotonic']==105
    clock[0]=101
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]
    assert q.observations((b'a',),'r')[0]['valid_until_monotonic']==105
    clock[0]=105.01
    assert q.observations((b'a',),'r')==[None]
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]


def test_known_new_source_incarnation_or_epoch_invalidates_cached_positive():
    for change in ('incarnation','epoch'):
        q,b,clock,messages,p=helper()
        q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
        q.receive(response(messages[-1][1]));q.poll(p)
        if change=='incarnation':b._dangling_ops.sources['tcp://source:1']='replacement'
        else:b._dangling_ops.source_progress_epochs['tcp://source:1','inc']=1
        assert q.observations((b'a',),'r')==[None]
        assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]


@pytest.mark.parametrize('mutation',[
    {'sender_control_endpoint':'tcp://wrong:1'},{'sender_incarnation':'other'},
    {'request_generation':'other'},{'keys':[b'b']},{'deadline':999},
    {'ready':[1]},{'generations':[None]},{'export_sequence':-1},
    {'requested_epoch':True},{'sender_progress_epoch':True},
    {'ready':[]},{'state':'unavailable','ready':[True]},
])
def test_malformed_stale_or_wrong_source_reply_never_authorizes(mutation):
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
    q.receive(response(messages[-1][1],**mutation));q.poll(p)
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]


def test_deadline_is_from_first_admission_and_late_reply_cannot_revive():
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',100);q.poll(p)
    wire=messages[-1][1]
    clock[0]=100.09
    q.query((b'a',),'r','tcp://source:1',5000)
    assert next(iter(q.active.values())).deadline==100.1
    clock[0]=100.11
    q.receive(response(wire));q.poll(p)
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.MISS,None)]


def test_busy_is_pending_and_stale_epoch_retries_without_deadline_refresh():
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',100);q.poll(p)
    wire=messages[-1][1]
    q.receive(response(wire,state='busy'));q.poll(p)
    assert q.observations((b'a',),'r')==[None]
    q.receive(response(wire,state='stale',sender_progress_epoch=1));q.poll(p)
    clock[0]=100.02;q.poll(p)
    assert messages[-1][1]['source_progress_epoch']==1
    assert messages[-1][1]['deadline']==100.1


def test_batches_concurrency_deadlines_and_no_reordered_keys():
    q,b,clock,messages,p=helper()
    keys=tuple(str(i).encode() for i in range(5000))
    q.query(keys,'r','tcp://source:1',100);q.poll(p)
    assert len(q.active)==16 and len(messages)==16
    assert all(len(m['keys'])<=256 for _,m in messages)
    wire=messages[0][1]
    q.receive(response(wire,keys=list(reversed(wire['keys']))));q.poll(p)
    assert len(q.active)==16
    clock[0]=100.11
    for _ in range(3):q.poll(p)
    assert all(s==QueryStatus.MISS for s,_ in q.query(keys,'r','tcp://source:1',5000))
    assert len(messages)==16 # queued work expired without sending or refreshed deadlines


@pytest.mark.parametrize('bound',['MAX_KEYS','MAX_REQUESTS','MAX_QUEUED'])
def test_explicit_bounds_and_shutdown(monkeypatch,bound):
    import kvcr.ready_query as module
    q,b,clock,messages,p=helper()
    monkeypatch.setattr(module,bound,1)
    q.query((b'a',),'r','tcp://source:1',5000)
    with pytest.raises(BufferError):q.query((b'b',),'other','tcp://source:1',5000)
    q.close();q.poll(p)
    assert not q.requests and not q.active and not q.queued and not q.incoming
    assert q.key_count==0 and not messages


@pytest.mark.parametrize('case',['retired','incomplete','layout','not_ready'])
def test_source_snapshot_requires_complete_nonretired_exact_layout(case):
    source,agent,control,pins,memory=_source()
    local=source._core._local_dram
    entry=local._exports[b'a']
    if case=='retired':entry.residency.retire_on_release=True
    elif case=='incomplete':local._exports[b'a']=replace(entry,refs=(None,))
    elif case=='layout':local._exports[b'a']=replace(entry,key_layout=('other',))
    else:
        from kvcr.local_dram import _LocalDramState
        entry.residency.state=_LocalDramState.FILLING
    assert local.probe_export_subset((b'a',),('',))[0]==[False]
    assert local._export_readers()==0


def test_query_ready_records_only_returned_confirmed_or_pending_statuses():
    source,target,left,right,agent,pins,memory=pair()
    counts=[]
    target._core._compact_capture=SimpleNamespace(query_counts=lambda agent,rows:counts.append(rows))
    result=target.query_ready([b'a',b'b'],'req')
    assert counts==[result] and all(s==QueryStatus.FETCHING for s,_ in result)
    target._core._compact_capture=None


def test_global_budget_admits_multiple_full_size_requests_without_queue_failure():
    q,b,clock,messages,p=helper()
    keys=tuple(str(i).encode() for i in range(65536))
    for request_id in ('first','second'):
        statuses=q.query(keys,request_id,'tcp://source:1',5000)
        assert all(status==QueryStatus.FETCHING for status,_ in statuses)
    assert q.key_count==131072
    q.poll(p)
    assert len(q.active)==16 and len(messages)==16


def test_ready_observation_budget_covers_maximum_five_group_model_context():
    from kvcr.ready_query import MAX_REQUEST_KEYS
    max_tokens = 1_048_576
    required = sum(max_tokens // size for size in (64, 64, 256, 8, 4))
    assert required == 430_080
    assert MAX_REQUEST_KEYS >= required


def test_source_restart_requires_fresh_handshake_without_refreshing_deadline():
    q,b,clock,messages,p=helper()
    invalidated=[];b._invalidate_control_peer=lambda endpoint:invalidated.append(endpoint)
    q.query((b'a',),'r','tcp://source:1',100);q.poll(p)
    old=messages[-1][1]
    q.receive(response(old,state='stale',sender_incarnation='replacement'));q.poll(p)
    assert not invalidated
    assert b._dangling_ops.sources['tcp://source:1']=='inc' # native state is untouched
    clock[0]=100.02;q.poll(p)
    assert messages[-1][1]['source_incarnation']=='replacement'
    assert messages[-1][1]['deadline']==100.1
    # A late prior-generation response cannot revert the current handshake.
    q.receive(response(old));q.poll(p)
    assert q.peers['tcp://source:1'][0]=='replacement'
    assert q.observations((b'a',),'r')==[None]


def test_probe_peer_discovery_is_bounded_and_does_not_modify_native_peer_state():
    q,b,clock,messages,p=helper()
    b._dangling_ops.sources.clear()
    q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
    q.receive(response(messages[-1][1],state='stale'));q.poll(p)
    assert not b._dangling_ops.sources
    assert q.peers['tcp://source:1'][0]=='inc'
    q.discard('r')
    assert not q.peers


def test_malformed_oversize_control_fields_are_not_retained():
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
    wire=response(messages[-1][1],request_generation='x'*10000)
    q.receive(wire)
    assert not q.incoming


def test_frame_byte_bound_with_large_keys_and_receive_budget():
    q,b,clock,messages,p=helper()
    keys=tuple(i.to_bytes(4,'little')+b'x'*4092 for i in range(256))
    q.query(keys,'r','tcp://source:1',5000);q.poll(p)
    assert len(messages)==16
    assert all(sum(map(len,m['keys']))<=65536 for _,m in messages)


def test_real_source_busy_then_ready_and_halt_is_never_fetchable():
    source,target,left,right,agent,pins,memory=pair()
    local=source._core._local_dram
    with local._export_lock:
        target.query_ready([b'a'],'req')
        until=time.monotonic()+.05
        while time.monotonic()<until:
            exchange(left,right)
            assert target.query_ready([b'a'],'req')==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]
            time.sleep(.001)
    assert confirmed(target,left,right,(b'a',))==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]
    target.discard_hint('req');target.submit_hint(_router_hint(left.endpoint,[97]),'req')
    source._core._remote_fw_dram._dangling_ops._source_stalled=True
    assert confirmed(target,left,right,(b'a',))==[(QueryStatus.MISS,None)]


def test_exact_membership_ignores_opportunistic_and_discard_never_authorizes_old_reply():
    source,target,left,right,agent,pins,memory=pair()
    assert target.query_ready([b'unauthorized'],'req')==[(QueryStatus.MISS,None)]
    target.query_ready([b'a'],'req')
    until=time.monotonic()+1
    while time.monotonic()<until:
        exchange(left,right)
        if target._core._remote_fw_dram._ready_queries.active:break
        time.sleep(.001)
    target.discard_hint('req')
    exchange(left,right)
    assert target.query_ready([b'a'],'req')==[(QueryStatus.MISS,None)]
    assert target.query_ready_observations([b'a'],'req')==[None]


def test_probe_preserves_concurrent_native_reader_claim():
    from test_guard_only_sources import _request
    from _kvcr_test_utils import _wait_until
    source,target,left,right,agent,pins,memory=pair()
    agent.state='PROC'
    before=len(agent.xfers)
    try:
        _request(left,[b'a'])
        _wait_until(lambda:len(agent.xfers)>before)
        readers=source._core._local_dram._export_readers()
        assert readers==1
        assert confirmed(target,left,right,(b'a',))==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]
        assert source._core._local_dram._export_readers()==readers
        assert len(agent.xfers)==before+1
    finally:
        agent.state='DONE'
        _wait_until(lambda:source._core._local_dram._export_readers()==0)


def test_full_pool_byte_geometry_is_required_even_for_matching_key():
    source,target,left,right,agent,pins,memory=pair()
    target._core.pool_layouts=[('',32)]
    assert confirmed(target,left,right,(b'a',))==[(QueryStatus.MISS,None)]


def test_duplicate_reply_does_not_extend_observation_and_discard_clears_all_metadata():
    q,b,clock,messages,p=helper()
    q.query((b'a',),'r','tcp://source:1',5000);q.poll(p)
    wire=response(messages[-1][1]);q.receive(wire);q.poll(p)
    clock[0]=101;q.receive(wire);q.poll(p)
    assert q.observations((b'a',),'r')[0]['valid_until_monotonic']==105
    q.discard('r')
    assert q.observations((b'a',),'r')==[None] and q.key_count==0


@pytest.mark.parametrize('timeout',[0,-1,True,5001,1.5])
def test_invalid_timeout_does_not_admit_pending_work(timeout):
    source,target,left,right,agent,pins,memory=pair()
    with pytest.raises(ValueError):target.query_ready([b'a'],'req',timeout_ms=timeout)
    assert not target._core._remote_fw_dram._ready_queries.requests


def test_actual_local_fill_pending_converges_to_ready_without_any_hint():
    from _kvcr_test_utils import _mem_descriptor,_poll_until
    source,agent,control,pins,memory=_source()
    agent.state='PROC'
    handle=source.deposit({b'b':[_mem_descriptor(1,end_point_name='source')]})
    try:
        assert source.query_ready([b'b'],'without-hint')==[(QueryStatus.FETCHING,CacheTier.LOCAL_G2)]
    finally:
        agent.state='DONE'
    assert _poll_until(source,bool)[0][0]==handle
    assert source.query_ready([b'b'],'without-hint')==[(QueryStatus.HIT,CacheTier.LOCAL_G2)]
    source._core._block_record_map[b'b'].local_dram.pending_labels.add('')
    assert source.query_ready([b'b'],'without-hint')!=[(QueryStatus.HIT,CacheTier.LOCAL_G2)]


@pytest.mark.parametrize('old_location',['queued','active'])
def test_expired_old_probe_cannot_finish_renewed_key_observation(monkeypatch,old_location):
    import kvcr.ready_query as module
    q,b,clock,messages,p=helper()
    monkeypatch.setattr(module,'CONCURRENT',0 if old_location=='queued' else 1)
    q.query((b'a',),'r','tcp://source:1',100);q.poll(p)
    generation=q.requests['r'].generation
    clock[0]=100.2
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.MISS,None)]
    clock[0]=105.21
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHING,CacheTier.REMOTE_G2)]
    assert q.requests['r'].generation==generation
    new_deadline=q.requests['r'].observations[b'a'].deadline
    monkeypatch.setattr(module,'CONCURRENT',1)
    q.poll(p) # expires the OLD queued/active probe after the key was renewed
    assert q.requests['r'].observations[b'a'].ready is None
    assert q.requests['r'].observations[b'a'].deadline==new_deadline
    q.poll(p)
    q.receive(response(messages[-1][1]));q.poll(p)
    assert q.query((b'a',),'r','tcp://source:1',5000)==[(QueryStatus.FETCHABLE,CacheTier.REMOTE_G2)]


def test_queue_bound_counts_actual_byte_limited_frames_before_admission(monkeypatch):
    import kvcr.ready_query as module
    q,b,clock,messages,p=helper()
    monkeypatch.setattr(module,'MAX_QUEUED',1)
    keys=tuple(i.to_bytes(4,'little')+b'x'*4092 for i in range(17))
    with pytest.raises(BufferError):q.query(keys,'r','tcp://source:1',5000)
    assert not q.queued and q.key_count==0
    assert not q.requests['r'].observations
